import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
import cv2
import os
import time

def get_model_instance_segmentation(num_classes):
    """Load instance segmentation model with custom number of classes"""
    # Load pre-trained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def load_trained_model(model_path, num_classes, device):
    """Load the trained model from saved weights"""
    model = get_model_instance_segmentation(num_classes)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found!")
        print("Please run train.py first to train and save the model.")
        return None

    model.to(device)
    model.eval()
    return model

def detect_persons(frame, model, device, threshold=0.5):
    """
    Detect persons in a frame using the trained model
    Args:
        frame: OpenCV image (BGR format)
        model: PyTorch detection model
        device: torch device
        threshold: confidence threshold for detections
    Returns:
        frame with bounding boxes and masks drawn
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to tensor and normalize
    img_tensor = F.to_tensor(img_rgb).to(device)

    # Perform inference
    with torch.no_grad():
        predictions = model([img_tensor])

    # Get predictions
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    masks = pred['masks'].cpu().numpy()

    # For our trained model: class 1 is person (class 0 is background)
    person_label = 1

    # Draw bounding boxes and masks for persons above threshold
    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if label == person_label and score > threshold:
            x1, y1, x2, y2 = box.astype(int)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw mask (optional, semi-transparent overlay)
            mask = masks[idx, 0]
            mask = mask > 0.5  # Threshold mask
            mask_resized = cv2.resize(mask.astype('uint8'),
                                     (frame.shape[1], frame.shape[0]))

            # Create colored mask overlay
            color_mask = frame.copy()
            color_mask[mask_resized == 1] = [0, 255, 0]
            frame = cv2.addWeighted(frame, 0.8, color_mask, 0.2, 0)

            # Add label with confidence score
            label_text = f'Person: {score:.2f}'
            cv2.putText(frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Model parameters (same as training)
    num_classes = 2  # background + person
    model_path = 'person_detection_model.pth'

    # Load trained model
    print("Loading trained model...")
    model = load_trained_model(model_path, num_classes, device)

    if model is None:
        print("Failed to load model. Exiting...")
        return

    print("Model loaded successfully!")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("\n" + "="*50)
    print("Real-time Person Detection with Trained Model")
    print("="*50)
    print("Press 'q' to quit")
    print("="*50 + "\n")

    frame_count = 0
    fps = 0
    fps_update_interval = 10  # Update FPS every N frames
    start_time = time.time()

    try:
        while True:
            # Start time for this frame
            frame_start_time = time.time()

            # Read frame from webcam
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame")
                break

            frame_count += 1

            # Detect persons in frame
            frame_with_detection = detect_persons(frame, model, device, threshold=0.7)

            # Calculate FPS
            frame_time = time.time() - frame_start_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0

            # Update average FPS periodically for smoother display
            if frame_count % fps_update_interval == 0:
                fps = current_fps
            else:
                # Simple moving average
                fps = fps * 0.9 + current_fps * 0.1

            # Add FPS and frame counter
            cv2.putText(frame_with_detection, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame_with_detection, f'Frame: {frame_count}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the frame
            cv2.imshow('Real-time Person Detection (Trained Model)', frame_with_detection)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        # Calculate statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print("\nDetection stopped")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
