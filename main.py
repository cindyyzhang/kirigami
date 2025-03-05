import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from jaxopt import LBFGS
import os
import time

from space_groups import PlaneGroup
from tile import Tile, TileConfig
from pattern import Pattern
from flow import FlowConfig, EquivariantFlow
from plot import plot_tiling

def apply_pattern_flow(pattern: Pattern, flow: EquivariantFlow, vector_field: callable) -> Pattern:
    """
    Applies diagonal flow slicing to entire pattern.
    """
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
            t = (d - cumdist[idx-1]) / (cumdist[idx] - cumdist[idx-1])
            point = (1 - t) * contour[idx - 1] + t * contour[idx]
            sampled_points.append(point)
    return np.array(sampled_points)

def get_boundary(mask, num_points=64):
    """
    Given a binary mask (2D array), extract the largest contour and sample 64 points along it.
    """
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
        
    # Select the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # If contour is too small, return None to avoid processing noise
    if cv2.contourArea(largest_contour) < 500:  # Minimum area threshold
        return None
        
    largest_contour = largest_contour.squeeze()
    # Sample 64 points along the contour
    points = sample_contour_points(largest_contour, num_points)
    return points

def preprocess_boundary_points(points):
    """
    Normalize and center the boundary points to prepare them for the flow algorithm.
    """
    if points is None:
        return None
    points = jnp.array(points)
    
    # Center the points around origin
    center = jnp.mean(points, axis=0)
    centered_points = points - center
    
    # Scale to similar range as the simulated circle (radius ~1)
    squared_distances = jnp.sum(centered_points**2, axis=1)
    max_dist = jnp.sqrt(jnp.max(squared_distances))
    scale_factor = jnp.where(max_dist > 0, 1.0 / max_dist, 1.0)
    normalized_points = centered_points * scale_factor
    return normalized_points

class FlowOptimizer:
    """Wrapper class to handle optimization with changing target points"""
    def __init__(self, flow, tile_points, init_params):
        self.flow = flow
        self.tile_points = jax.device_put(tile_points)
        self.times = jax.device_put(
            jnp.linspace(flow.config.start_time, flow.config.duration, flow.config.num_steps)
        )
        
        @jax.jit
        def vector_field_param(params, xy, t):
            op_uv = jax.vmap(
                lambda xy: jax.vmap(
                    lambda op: flow._apply_operation(flow.base_function, params, xy, t, op)
                )(flow.operations)
            )(xy)
            return jnp.mean(op_uv, axis=1)
        
        @jax.jit
        def flow_loss(params, target_points):
            # Push points forward in time
            transformed = jax.experimental.ode.odeint(
                lambda xy, t: vector_field_param(params, xy, t),
                self.tile_points,
                self.times
            )
            final_points = transformed[-1]
            
            # Scale and center
            centroid_T = jnp.mean(final_points, axis=0)
            centroid_P = jnp.mean(target_points, axis=0)
            scale_T = jnp.mean(jnp.linalg.norm(final_points - centroid_T, axis=1))
            scale_P = jnp.mean(jnp.linalg.norm(target_points - centroid_P, axis=1))
            T_scaled = (final_points - centroid_T) * (scale_P / scale_T) + centroid_P
            
            # Compute costs for all possible cyclic shifts
            def compute_shift_cost(shift):
                shifted_target = jnp.roll(target_points, shift, axis=0)
                return jnp.sum((T_scaled - shifted_target) ** 2)
            
            shifts = jnp.arange(len(target_points))
            all_costs = jax.vmap(compute_shift_cost)(shifts)
            return jnp.min(all_costs)
            
        self.loss_fn = flow_loss
        self.current_target = None
        
        def opt_wrapper(params):
            dummy_target = jnp.zeros((64, 2))
            target = self.current_target if self.current_target is not None else dummy_target
            return self.loss_fn(params, target)
        
        self.optimizer = LBFGS(fun=opt_wrapper, maxiter=50, tol=1e-5)
        self.opt_state = self.optimizer.init_state(init_params)
        self.params = init_params
        self.initial_params = init_params
    
    def update(self, target_points):
        """Perform one optimization step with new target points"""
            
        self.current_target = target_points
        old_params = self.params

        self.params, self.opt_state = self.optimizer.update(
            self.params,
            self.opt_state
        )
        param_diff = jnp.abs(self.params - old_params).mean()
        loss = self.loss_fn(self.params, target_points)
        
        # jax.debug.print("Parameter change: {diff:.6f}", diff=param_diff)
        # jax.debug.print("Current loss: {loss:.4f}", loss=loss)
            
        return self.params, loss, param_diff

    def reset_optimizer(self):
        """
        Resets the LBFGS optimizer state with the initial parameters.
        """
        self.params = self.initial_params
        self.opt_state = self.optimizer.init_state(self.params)

def prevent_display_sleep():
    """Prevent the display from going to sleep."""
    try:
        os.system('xset s off -dpms')
        print("Display sleep prevention enabled on Linux")
    except Exception as e:
        print(f"Failed to prevent display sleep on Linux: {e}")

def main():
    # Define configurations for the flow and the tile.
    flow_config = FlowConfig(
        num_feats=50,
        lengthscale=0.8,
        timescale=1.0,
        amplitude=0.4,
        duration=1.0,
        num_steps=100,
        start_time=0.01
    )
    tile_config = TileConfig(
        num_points=65,
        boundary_offset=0.1,
        inner_offset=0.2
    )
    group = PlaneGroup(4) # pg
    flow = EquivariantFlow(group, flow_config)
    
    # Generate a tile from the fundamental region and create a tiling pattern
    tile = Tile.from_fundamental_region(group, tile_config)
    tile_points = tile.boundary
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
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Starting webcam segmentation.")
    
    # Set up display window for the flow pattern
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    plt.ion()

    # Initialize flow optimizer
    flow_optimizer = FlowOptimizer(flow, tile_points, flow.params)
    
    # Variables for frame skipping and processing
    frame_count = 0
    process_every_n_frames = 2  # Process every 2nd frame for segmentation
    last_mask = None
    last_boundary_points = None
    mask_overlay = None
    
    # Variables to track optimization progress
    optimization_steps = 0
    reset_every_n_steps = 50  # Reset optimizer every 50 steps
    consecutive_small_changes = 0
    small_change_threshold = 0.001  # Parameter change threshold
    max_consecutive_small_changes = 10  # Reset after this many small changes
    
    prevent_display_sleep()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            # Add retry logic for webcam failures
            cap.release()
            time.sleep(5)  # Wait 5 seconds before trying to reconnect
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        
        # Only process every n frames for segmentation
        process_this_frame = (frame_count % process_every_n_frames == 0)
        frame_count += 1
        
        if process_this_frame:
            # Convert to RGB and process with model
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image).to(device)
            
            with torch.no_grad():
                predictions = model([image_tensor])
            prediction = predictions[0]
            
            # Filter out detections for persons
            person_indices = [i for i, label in enumerate(prediction['labels'].cpu().numpy()) 
                            if label == 1 and prediction['scores'][i] > 0.7]
            
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
                
                # Get mask and update last_mask
                last_mask = prediction['masks'][main_person_index, 0].cpu().numpy() > 0.5
                
                # Extract boundary points (only when processing a frame)
                if last_mask is not None:
                    last_boundary_points = get_boundary(last_mask)
            else:
                last_mask = None
                last_boundary_points = None
        
        # Always display the most recent mask (even on frames we don't process)
        if last_mask is not None:
            if mask_overlay is None or mask_overlay.shape != frame.shape:
                mask_overlay = np.zeros_like(frame, dtype=np.uint8)
            else:
                mask_overlay.fill(0)  # Clear the buffer
                
            mask_overlay[last_mask] = (0, 255, 0)  # Green color in BGR
            blended = cv2.addWeighted(frame, 1.0, mask_overlay, 0.5, 0)
        else:
            blended = frame
        

        if process_this_frame and last_boundary_points is not None:
            person_points = preprocess_boundary_points(last_boundary_points)
            
            if person_points is not None:
                # Update the flow optimizer with the new target points
                params, loss, param_diff = flow_optimizer.update(person_points)
                optimization_steps += 1

                if param_diff < small_change_threshold:
                    consecutive_small_changes += 1
                else:
                    consecutive_small_changes = 0
                
                # Reset optimizer if:
                # 1. We've done reset_every_n_steps optimization steps, or
                # 2. We've had max_consecutive_small_changes small parameter changes
                if (optimization_steps % reset_every_n_steps == 0 or 
                    consecutive_small_changes >= max_consecutive_small_changes):
                    print(f"Resetting optimizer. Steps: {optimization_steps}, Loss: {loss:.4f}")
                    flow_optimizer.reset_optimizer()
                    consecutive_small_changes = 0
                
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

if __name__ == "__main__":
    main()