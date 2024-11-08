import cv2
import json
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import ops

# Load parking lots from JSON
with open('parking_lots.json', 'r') as f:
    parking_lots = json.load(f)

# Initialize YOLOv8 model
model = torch.hub.load('ultralytics/yolo', 'yolov8l.pt')  # Change to your model
model.eval()

# Open video
video_path = 'your_video.mp4'  # Replace with your video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)
    detections = results.xyxy[0]  # Get detections

    # Draw parking lot boxes
    for lot in parking_lots:
        x1, y1, x2, y2 = lot["coordinates"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
        cv2.putText(frame, lot["name"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check vacancy
        lot["vacant"] = True  # Assume vacant
        for det in detections:
            if (det[0] >= x1 and det[1] >= y1 and det[2] <= x2 and det[3] <= y2):  # Check overlap
                lot["vacant"] = False  # Not vacant
                break

    # Display frame
    cv2.imshow('Video', frame)

    # Update the database based on vacancy (implement your database logic here)
    # Example: update_database(lot["name"], lot["vacant"])

    if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
