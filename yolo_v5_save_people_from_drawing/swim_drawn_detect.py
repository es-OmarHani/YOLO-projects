import cv2
import torch
from tracker import Tracker
import numpy as np

# Load YOLOv5 model
yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open video file for tracking
video_path = 'videos/drawn_person.mp4'
video_capture = cv2.VideoCapture(video_path)

# Create object tracker
object_tracker = Tracker()

# Main loop for video processing
while True:
    # Read a frame from the video
    ret, current_frame = video_capture.read()
    if not ret:
        break

    # Resize the frame for better processing
    resized_frame = cv2.resize(current_frame, (1020, 500))

    # Run object detection with YOLOv5
    detection_results = yolov5_model(resized_frame)

    # Extract bounding boxes of detected persons
    detected_persons = []
    for index, row in detection_results.pandas().xyxy[0].iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        class_name = row['name']
        
        if 'person' in class_name:
            detected_persons.append([x1, y1, x2, y2])

    # Update the object tracker with detected persons' bounding boxes
    tracked_boxes_ids = object_tracker.update(detected_persons)

    # Display information and draw rectangles on the frame
    num_detected_persons = len(detected_persons)
    for tracked_box_id in tracked_boxes_ids:
        x, y, w, h, person_id = tracked_box_id

        # Draw rectangle around the person
        cv2.rectangle(resized_frame, (x, y), (w, h), (0, 255, 0), 2)

        # Display person ID
        cv2.putText(resized_frame, 'Person ID=' + str(person_id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    # Display the number of detected persons
    cv2.putText(resized_frame, 'Number of Persons=' + str(num_detected_persons), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)

    # Show the frame
    cv2.imshow('FRAME', resized_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
