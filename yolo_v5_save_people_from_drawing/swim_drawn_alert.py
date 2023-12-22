import cv2
import torch
from tracker import Tracker
import numpy as np
import mediapipe as mp
import playsound
from mutagen.mp3 import MP3
import time
import miniaudio

# Initialize Mediapipe Pose module
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Path to the alarm sound file
ALARM_PATH = "audios/accident.mp3"

# Load YOLOv5 model
yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open video file for processing
cap = cv2.VideoCapture('videos/drawn_person.mp4')

# Initialize object tracker
tracker = Tracker()

# # Set up alarm sound file
# file = 'audios/assets_alarm.mp3'
# audio = MP3(file)
# length = audio.info.length

# Frame check for alarm triggering
frame_check = 7
flag = 0


import numpy as np

def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle formed by three points.

    Args:
        point_a (list): Coordinates of the first point [x, y].
        point_b (list): Coordinates of the second point [x, y].
        point_c (list): Coordinates of the third point [x, y].

    Returns:
        float: The calculated angle in degrees.
    """
    # Convert points to NumPy arrays for vector operations
    a = np.array(point_a)
    b = np.array(point_b)
    c = np.array(point_c)

    # Calculate angles using arctangent and convert to degrees
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180 / np.pi)

    # Ensure the angle is within the range [0, 180]
    if angle > 180.0:
        angle = 360 - angle

    return angle


# Main loop for video processing
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv5 object detection
    results = yolov5_model(frame)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply the Mediapipe Pose detection module
    result = pose.process(imgRGB)

    # Extract from frame some values
    h, w, c = frame.shape

    # Draw landmarks on the frame
    if result.pose_landmarks:
        mpDraw.draw_landmarks(frame, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark

        # Extracting landmarks for angle calculation
        l_shoulder = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]

        r_shoulder = [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].y]

        # Calculate angles and display on the frame
        l_ang = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_ang = calculate_angle(r_shoulder, r_elbow, r_wrist)

        cv2.putText(frame, str(int(l_ang)), tuple(np.multiply(l_elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
        cv2.putText(frame, str(int(r_ang)), tuple(np.multiply(r_elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

        # Check for warning conditions and trigger alarm
        if l_wrist[1] * h < l_elbow[1] * h < l_shoulder[1] * h and l_ang > 150:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, 'Warning! Someone needs help', (20, 75), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                # stream = miniaudio.stream_file(file)
                # with miniaudio.PlaybackDevice() as device:
                #     device.start(stream)
                #     time.sleep(length)

        elif r_wrist[1] * h < r_elbow[1] * h < r_shoulder[1] * h and r_ang > 150:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, 'Warning!!! Someone needs help', (20, 75), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                # stream = miniaudio.stream_file(file)
                # with miniaudio.PlaybackDevice() as device:
                #     device.start(stream)
                #     time.sleep(length)

    # Process YOLOv5 detection results
    points = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        n = row['name']

        # Filter out low-confidence detections
        if 'person' in n and row['confidence'] > 0.25:
            points.append([x1, y1, x2, y2])

    # Update object tracker with detected persons' bounding boxes
    boxes_id = tracker.update(points)

    num = len(points)
    id = []
    for box_id in boxes_id:
        x, y, w, h, idd = box_id
        id.append(idd)

        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, 'number of persons=' + str(num), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
        cv2.putText(frame, 'person ID=' + str(id[-1]), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    cv2.imshow('FRAME', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
