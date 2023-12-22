import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('videos/1.mp4')

# Load the YOLO model from the ONNX file
net = cv2.dnn.readNetFromONNX("yolov5n.onnx")

# Load the COCO class names
file = open("coco.txt", "r")
classes = file.read().split('\n')
print(classes)

while True:
    # Read a frame from the video
    img = cap.read()[1]

    # Break the loop if the video is finished
    if img is None:
        break

    # Resize the image to a fixed size (1000x600)
    img = cv2.resize(img, (1000, 600))

    # Prepare the image for YOLO model
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(640, 640), mean=[0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)

    # Get the detections from the YOLO model
    detections = net.forward()[0]

    # Initialize lists to store class IDs, confidences, and bounding boxes
    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    # Calculate scaling factors for bounding box coordinates
    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width / 640
    y_scale = img_height / 640

    # Iterate through the detected rows
    for i in range(rows):
        row = detections[i]
        confidence = row[4]

        # Check if the confidence is above a threshold (0.5)
        if confidence > 0.5:
            classes_score = row[5:]
            ind = np.argmax(classes_score)

            # Check if the class score is above a threshold (0.5)
            if classes_score[ind] > 0.5:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]

                # Scale the bounding box coordinates
                x1 = int((cx - w/2) * x_scale)
                y1 = int((cy - h/2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)

                box = np.array([x1, y1, width, height])
                boxes.append(box)

    # Apply non-maximum suppression to eliminate redundant bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    # Draw bounding boxes and labels on the image
    for i in indices:
        x1, y1, w, h = boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = f"{label} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255, 0, 0), 1)
        cv2.putText(img, text, (x1, y1-2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 1)

    # Display the annotated image
    cv2.imshow("VIDEO", img)

    # Check for the 'q' key to exit the loop
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
