import cv2
import torch
import numpy as np
from tracker import Tracker

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) 

# Open video capture
video_cap = cv2.VideoCapture('videos/highway.mp4')

# Initialize object tracker
object_tracker = Tracker()

# Callback function for mouse events
def handle_mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = frame[y, x]  # Get the pixel value at the clicked point
        print(f"Clicked at ({x}, {y}) - Pixel Value: {pixel_value}")

# Create a window and set mouse callback
cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', handle_mouse_event)

# Define regions of interest
green_region = [(611, 323), (666, 411), (795, 391), (706, 303)]
# yellow_region = [(710, 429), (724, 442), (775, 434), (769, 419)]

# Initialize sets to track unique IDs in each region
green_region_ids = set()
# yellow_region_ids = set()

while True:
    # Read a frame from the video
    ret, frame = video_cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (1020, 600))

    # Draw polygons for regions of interest
    cv2.polylines(frame, [np.array(green_region, np.int32)], True, (0, 255, 0), 2)
    # cv2.polylines(frame, [np.array(yellow_region, np.int32)], True, (0, 255, 255), 2)

    # Run YOLOv5 model on the frame
    results = yolo_model(frame)

    # Extract bounding boxes of cars
    car_bounding_boxes = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        name = row['name']
        if 'car' in name:
            car_bounding_boxes.append([x1, y1, x2, y2])

    # Update object tracker with car bounding boxes
    tracked_boxes_ids = object_tracker.update(car_bounding_boxes)
    # print(tracked_boxes_ids)

    # Process tracked boxes and IDs
    for box_id in tracked_boxes_ids:
        x, y, w, h, obj_id = box_id
        x, y, w, h, obj_id = int(x), int(y), int(w), int(h), int(obj_id)
        
        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 1)
        cv2.putText(frame, str(obj_id), (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 1)
        cv2.circle(frame, (w, h), 4, (0, 255, 0), -1)

        # Check if the object is in the green or yellow region
        green_region_result = cv2.pointPolygonTest(np.array(green_region, np.int32), (w, h), False)
        # yellow_region_result = cv2.pointPolygonTest(np.array(yellow_region, np.int32), (w, h), False)

        # Update sets based on region results
        if green_region_result >= 0:
            green_region_ids.add(obj_id)

            # cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 1)
            # cv2.putText(frame, str(obj_id), (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 1)
            # cv2.circle(frame, (w, h), 4, (0, 255, 0), -1)
        
        # if yellow_region_result >= 0:
        #     yellow_region_ids.add(obj_id)

    # Display the number of cars in each region
    num_cars_green = len(green_region_ids)
    # num_cars_yellow = len(yellow_region_ids)
    cv2.putText(frame, f'Number of cars in the green region: {num_cars_green}', (50, 65),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    # cv2.putText(frame, f'Number of cars in the yellow region: {num_cars_yellow}', (50, 90),
    #             cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    # Display the frame
    cv2.imshow("FRAME", frame)

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and close all windows
video_cap.release()
cv2.destroyAllWindows()
