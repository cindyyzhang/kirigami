import jax
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from jaxopt import LBFGS

from space_groups import PlaneGroup
from tile import Tile, TileConfig
from pattern import Pattern
from flow import FlowConfig, EquivariantFlow
from plot import plot_tiling

def apply_pattern_flow(pattern: Pattern, flow: EquivariantFlow, vector_field: callable) -> Pattern:
    """
    Applies diagonal flow slicing to entire pattern.
    
    Args:
        pattern: The input pattern
        flow: EquivariantFlow instance
        vector_field: Vector field to integrate
        
    Returns:
        New pattern with transformed vertices
    """
    # Get all vertices and their tile associations
    all_vertices, tile_indices = pattern.get_all_vertices()
    
    # Apply diagonal flow to all vertices at once
    transformed_vertices = flow.integrate_left_right(vector_field, all_vertices)
    
    # Create new pattern and add new tiles
    new_pattern = Pattern(pattern.group, pattern.width, pattern.height)
    for idx in range(len(transformed_vertices)):
        new_tile = Tile(transformed_vertices[idx], boundary_offset=0.1, inner_offset=0.1)
        new_pattern.add_tile(new_tile)
    return new_pattern

def sample_contour_points(contour, num_points=64):
    """
    Given a contour (an array of shape (N,2)), sample num_points evenly spaced along its arc length.
    """
    # Compute the distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    # Compute the cumulative distance along the contour
    cumdist = np.concatenate(([0], np.cumsum(distances)))
    total_length = cumdist[-1]
    # Create equally spaced sample distances
    sample_distances = np.linspace(0, total_length, num_points, endpoint=False)
    sampled_points = []
    # For each desired distance, find its location on the contour
    for d in sample_distances:
        idx = np.searchsorted(cumdist, d)
        if idx == 0:
            sampled_points.append(contour[0])
        else:
            # Interpolate linearly between contour[idx-1] and contour[idx]
            t = (d - cumdist[idx-1]) / (cumdist[idx] - cumdist[idx-1])
            point = (1 - t) * contour[idx - 1] + t * contour[idx]
            sampled_points.append(point)
    return np.array(sampled_points)

def get_boundary(mask, num_points=64):
    """
    Given a binary mask (2D array), extract the largest contour and sample 64 points along it.
    """
    # Convert boolean mask to uint8 image (values 0 or 255)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    # Select the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour = largest_contour.squeeze()  # shape (N,2)
    if largest_contour.ndim == 1:
        largest_contour = largest_contour[np.newaxis, :]
    # Sample 64 evenly spaced points along the contour
    points = sample_contour_points(largest_contour, num_points)
    return points

def preprocess_boundary_points(points):
    """
    Normalize and center the boundary points to prepare them for the flow algorithm.
    
    Args:
        points: NumPy array of boundary points from webcam segmentation
        
    Returns:
        JAX array of normalized points
    """
    if points is None:
        return None
    
    # Center the points around origin
    center = np.mean(points, axis=0)
    centered_points = points - center
    
    # Scale to similar range as the simulated circle (radius ~1)
    max_dist = np.max(np.sqrt(np.sum(centered_points**2, axis=1)))
    normalized_points = centered_points / max_dist if max_dist > 0 else centered_points
    
    # Convert to JAX array
    return jnp.array(normalized_points)

def flow_loss(params, tile_points, person_points, flow):
    """
    Compute loss between transformed tile points and target person points,
    considering all possible cyclic shifts of the points.
    
    Args:
        params: Flow parameters to optimize
        tile_points: Array of shape (64,2) representing the tile's border points
        person_points: Array of shape (64,2) representing the person outline
        flow: EquivariantFlow instance
    
    Returns:
        Loss value measuring minimum discrepancy over all possible shifts
    """
    def vector_field_param(xy, t, params):
        op_uv = jax.vmap(
            lambda op: flow._apply_operation(flow.base_function, params, xy, t, op)
        )(flow.operations)
        return jnp.mean(op_uv, axis=0)
    
    # Integrate tile points with current parameters
    times = jnp.linspace(0, flow.config.duration, flow.config.num_steps)
    transformed = jax.experimental.ode.odeint(
        lambda xy, t: vector_field_param(xy, t, params),
        tile_points[jnp.newaxis, :, :],
        times
    )
    T = transformed[-1][0]  # Final transformed points
    
    # Compute centroids and average scales
    centroid_T = jnp.mean(T, axis=0)
    centroid_P = jnp.mean(person_points, axis=0)
    scale_T = jnp.mean(jnp.linalg.norm(T - centroid_T, axis=1))
    scale_P = jnp.mean(jnp.linalg.norm(person_points - centroid_P, axis=1))
    scale_ratio = scale_P / scale_T
    
    # Rescale transformed tile to match person's scale and centroid
    T_scaled = (T - centroid_T) * scale_ratio + centroid_P
    
    # Compute costs for all possible shifts
    def compute_shift_cost(shift):
        shifted_person = jnp.roll(person_points, shift, axis=0)
        return jnp.sum((T_scaled - shifted_person) ** 2)
    
    # Vectorize over all possible shifts
    shifts = jnp.arange(len(person_points))
    all_costs = jax.vmap(compute_shift_cost)(shifts)
    
    # Return minimum cost over all shifts
    return jnp.min(all_costs)

class FlowOptimizer:
    """Wrapper class to handle optimization with changing target points"""
    def __init__(self, flow, tile_points, init_params):
        self.flow = flow
        self.tile_points = tile_points
        self.current_target = None
        
        # Create a jitted loss function that takes target points as state
        def loss_with_state(params, target_points):
            return flow_loss(params, tile_points, target_points, flow)
        self.loss_fn = jax.jit(loss_with_state)
        
        # Create optimizer with a wrapper function that uses the current target
        def opt_wrapper(params):
            # Use a dummy target for initialization
            dummy_target = jnp.zeros((64, 2))  # Assuming 64 points in 2D
            target = self.current_target if self.current_target is not None else dummy_target
            return self.loss_fn(params, target)
        
        # Create optimizer without state parameter
        self.optimizer = LBFGS(
            fun=opt_wrapper,
            maxiter=1,  # Changed from 100 to 1
            tol=1e-6,
            maxls=4,  # Limit line search steps
            history_size=10  # Smaller history size for more frequent updates
        )
        # Initialize optimizer state
        self.opt_state = self.optimizer.init_state(init_params)
        self.params = init_params
        
        # Store initial parameters for debugging
        self.initial_params = init_params
    
    def update(self, target_points):
        """Perform one optimization step with new target points"""
        # Update current target
        self.current_target = target_points
        
        # Store old parameters for comparison
        old_params = self.params
        
        # Do optimization step
        self.params, self.opt_state = self.optimizer.update(
            self.params,
            self.opt_state
        )
        
        # Compute parameter change for debugging
        param_diff = jnp.abs(self.params - old_params).mean()
        
        # Compute loss with current parameters and target
        loss = self.loss_fn(self.params, target_points)
        
        # Debug prints
        jax.debug.print("Parameter change: {diff:.6f}", diff=param_diff)
        jax.debug.print("Current loss: {loss:.4f}", loss=loss)
        
        return self.params, loss

def main():
    # Define configurations for the flow and the tile.
    flow_config = FlowConfig(
        num_feats=50,
        lengthscale=1.0,
        timescale=1.0,
        amplitude=0.6,
        duration=1.0,
        num_steps=100,
        start_time=0.01
    )
    tile_config = TileConfig(
        num_points=65,
        boundary_offset=0.1,
        inner_offset=0.2
    )
    
    # Create a plane group
    group = PlaneGroup(2)
    
    # Create an instance of EquivariantFlow
    flow = EquivariantFlow(group, flow_config)
    
    # Generate a tile from the fundamental region
    tile = Tile.from_fundamental_region(group, tile_config)
    tile_points = tile.boundary
    
    # Create a pattern with the original tile
    pattern = Pattern(group, width=6, height=4)
    pattern.add_tile(tile)
    pattern.generate_tiling()
    
    # Set up the webcam and segmentation model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device)
    model.eval()
    transform = T.Compose([T.ToTensor()])
    
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)
    print("Starting webcam segmentation. Press 'q' to exit.")
    
    # Set up display window for the flow pattern
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.ion()  # Turn on interactive mode for live updates

    # Initialize flow optimizer
    flow_optimizer = FlowOptimizer(flow, tile_points, flow.params)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert frame from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transform the image and move it to the device
        image_tensor = transform(image).to(device)
        
        # Perform inference
        with torch.no_grad():
            predictions = model([image_tensor])
        prediction = predictions[0]
        
        # Filter out detections for persons
        person_indices = [i for i, label in enumerate(prediction['labels'].cpu().numpy()) 
                          if label == 1 and prediction['scores'][i] > 0.7]
        
        mask = None
        if person_indices:
            # Choose the person with the largest mask area
            max_area = 0
            main_person_index = person_indices[0]
            for i in person_indices:
                mask_i = prediction['masks'][i, 0].cpu().numpy()
                binary_mask = mask_i > 0.5
                area = np.sum(binary_mask)
                if area > max_area:
                    max_area = area
                    main_person_index = i
            
            mask = prediction['masks'][main_person_index, 0].cpu().numpy() > 0.5
            
            # Create a colored overlay for the mask on the original frame
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[mask] = (0, 255, 0)  # Green color in BGR
            blended = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)
        else:
            blended = frame
        
        # Extract and process boundary points when a mask is available
        if mask is not None:
            boundary_points = get_boundary(mask)
            if boundary_points is not None:
                person_points = preprocess_boundary_points(boundary_points)
                
                _, loss = flow_optimizer.update(person_points)
                jax.debug.print("loss: {x:.4f}", x=loss)
                learned_vector_field = flow.get_equivariant_field(flow.base_function, flow_optimizer.params)
                transformed_pattern = apply_pattern_flow(pattern, flow, learned_vector_field)
                
                # Clear and reuse the same axes
                ax.clear()
                plot_tiling(transformed_pattern, ax=ax)
                fig.canvas.draw()
                plt.pause(0.01)  # Small pause to allow GUI to update
        
        # Display the webcam feed with points
        cv2.imshow("Webcam Segmentation", blended)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close('all')

if __name__ == "__main__":
    main()
