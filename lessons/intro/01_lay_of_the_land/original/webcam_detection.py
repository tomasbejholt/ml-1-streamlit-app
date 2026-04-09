"""
Live Object Detection with Webcam
Run this script to see real-time object detection using your webcam.
Press 'q' to quit.

Usage: python webcam_detection.py
"""
import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# Load model
print("Loading Faster R-CNN model...")
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
COCO_LABELS = weights.meta["categories"]
print(f"Ready! Detecting {len(COCO_LABELS)} object categories. Press 'q' to quit.")

# Colors for different object types
np.random.seed(42)
COLORS = {label: tuple(int(c) for c in np.random.randint(50, 255, 3)) for label in COCO_LABELS}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

to_tensor = transforms.ToTensor()
confidence_threshold = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) to RGB (PyTorch) and run detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = to_tensor(rgb).unsqueeze(0)

    with torch.no_grad():
        preds = model(img_tensor)[0]

    # Draw detections
    for i in range(len(preds['scores'])):
        score = preds['scores'][i].item()
        if score < confidence_threshold:
            continue

        label = COCO_LABELS[preds['labels'][i]]
        box = preds['boxes'][i].numpy().astype(int)
        color = COLORS[label]

        # Draw box and label
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        text = f"{label} {score:.0%}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (box[0], box[1] - text_size[1] - 8),
                     (box[0] + text_size[0], box[1]), color, -1)
        cv2.putText(frame, text, (box[0], box[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Object Detection (press q to quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
