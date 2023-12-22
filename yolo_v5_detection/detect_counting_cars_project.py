# import dependencies
import cv2 as cv
import torch
import numpy as np

def handle_mouse_click(event, x, y, flags, param):
    """
    Callback function to handle mouse click events and display pixel numbers on the image.
    
    Parameters:
    - event: Type of mouse event (e.g., cv.EVENT_LBUTTONDOWN for left mouse button click).
    - x, y: Coordinates of the mouse cursor.
    - flags, param: Additional parameters for the callback.
    """
    if event == cv.EVENT_LBUTTONDOWN:
        pixel_number_text = f"Pixel ({x}, {y})"
        cv.putText(frame, pixel_number_text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        cv.imshow("FRAME", frame)

        # Add the clicked point to the parking_area list
        parking_area.append((x, y))


# Create a window for displaying frames
cv.namedWindow('FRAME')
# Uncomment the line below to enable the mouse callback
cv.setMouseCallback('FRAME', handle_mouse_click)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Video settings
fourcc = cv.VideoWriter_fourcc(*'XVID')  # Use any fourcc type to improve quality for the saved video
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Open the video file
cap = cv.VideoCapture('parking_video.mp4')

# Define polygonal area for parking space
parking_area = []

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv.resize(frame, (1020, 600))

    # Perform object detection using YOLOv5
    results = model(frame)

    points = []  # List to store centroid points of detected cars within the parking area
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        object_class  = (row['name'])
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2

        # Check if the detected object is a car and if it is within the defined parking area
        if 'car' in object_class :
            results = cv.pointPolygonTest(np.array(parking_area, np.int32), (cx, cy), False)
            if results >= 0:
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv.putText(frame, str(object_class), (x1, y1), cv.FONT_HERSHEY_PLAIN, 2, (150, 150, 150), 1)
                points.append([cx])

    # Display the parking area and the number of cars within it
    cv.polylines(frame, [np.array(parking_area, np.int32)], True, (0, 255, 0), 2)
    cv.putText(frame, 'Number of cars in parking =' + str(len(points)), (50, 80), cv.FONT_HERSHEY_PLAIN, 2, (150, 150, 170), 1)

    # Show the frame
    cv.imshow("FRAME", frame)
    
    # Write the frame to the output video file
    out.write(frame)

    # Break the loop if 'Esc' key is pressed
    if cv.waitKey(1) == 27:
        break

# Release the video capture and video writer objects
cap.release()
out.release()

# Close all windows
cv.destroyAllWindows()
