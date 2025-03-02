import torch
import torchvision
import cv2
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import time

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.to(device)
model.eval()
transform = T.Compose([T.ToTensor()])

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

print("Starting webcam segmentation.")
current_points = None
start_time = time.time()
points_saved = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame from BGR (OpenCV default) to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Transform the image and move it to the device
    image_tensor = transform(image).to(device)
    
    # Perform inference (the model expects a list of images)
    with torch.no_grad():
        predictions = model([image_tensor])
    prediction = predictions[0]

    # Filter out detections for persons (in COCO, the 'person' label is 1)
    # Also, use a score threshold to filter out weak detections (here, 0.7)
    person_indices = [i for i, label in enumerate(prediction['labels'].cpu().numpy()) 
                      if label == 1 and prediction['scores'][i] > 0.7]

    if person_indices:
        # Choose the person with the largest mask area (to avoid getting people in the background)
        max_area = 0
        main_person_index = person_indices[0]
        for i in person_indices:
            mask = prediction['masks'][i, 0].cpu().numpy()
            # Binarize the mask with a threshold of 0.5
            binary_mask = mask > 0.5
            area = np.sum(binary_mask)
            if area > max_area:
                max_area = area
                main_person_index = i

        # Retrieve the binary mask for the main person
        mask = prediction['masks'][main_person_index, 0].cpu().numpy() > 0.5

        # Create a colored overlay (green) for the mask
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[mask] = (0, 255, 0)  # Green color in BGR

        # Blend the original frame with the overlay
        blended = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)
        
        if mask is not None:
            points = get_boundary(mask)
            current_points = points  # Store the current points
            # Plot points on the frame
            for pt in points:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(blended, (x, y), 3, (0, 0, 255), -1)
    else:
        blended = frame
        current_points = None  # No points to save if no person detected
    
    # Display the result
    cv2.imshow("Webcam Segmentation", blended)
    
    # Check if 10 seconds have passed and we haven't saved points yet
    elapsed_time = time.time() - start_time
    if elapsed_time >= 10 and not points_saved and current_points is not None:
        np.save("points.npy", current_points)
        print(f"Points saved to points.npy after {elapsed_time:.2f} seconds")
        points_saved = True
    
    key = cv2.waitKey(1) & 0xFF
    # Add option to quit with 'q' key
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Application closed successfully.")