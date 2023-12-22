# import cv2 as cv
# from simple_face_recognition import SimpleFaceRecognition

# # Initialize SimpleFaceRecognition for face encoding and detection
# sfr = SimpleFaceRecognition()
# sfr.load_encoding_images("images/train/")  # Load face encodings from the specified folder

# # Open the camera
# cap = cv.VideoCapture(0)

# while True:
#     # Capture a frame from the camera
#     ret, frame = cap.read()

#     # Detect known faces in the captured frame
#     face_locations, face_names = sfr.detect_known_faces(frame)

#     # Draw rectangles and labels around detected faces
#     for face_loc, name in zip(face_locations, face_names):
#         y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

#         # Display the name above the face
#         cv.putText(frame, name, (x1, y1 - 10), cv.FONT_HERSHEY_DUPLEX, 1, (170, 170, 200), 1)
        
#         # Draw a rectangle around the face
#         cv.rectangle(frame, (x1, y1), (x2, y2), (170, 170, 170), 1)

#     # Display the frame with face recognition results
#     cv.imshow("Frame", frame)

#     # Check for the 'Esc' key to exit the loop
#     key = cv.waitKey(1)
#     if key == 27:
#         break

# # Release the camera and close all windows
# cap.release()
# cv.destroyAllWindows()


import cv2 as cv
from simple_face_recognition import SimpleFaceRecognition

# Initialize SimpleFaceRecognition for face encoding and detection
sfr = SimpleFaceRecognition()
sfr.load_encoding_images("images/train/")  # Load face encodings from the specified folder

# Read an example image
image_path = "images/test/4.jpg"
frame = cv.imread(image_path)
factor = 0.5

# Resize the frame
width = int(frame.shape[1] * factor)
height = int(frame.shape[0] * factor)
dim = (width, height)
small_frame = cv.resize(frame, dim, interpolation = cv.INTER_AREA)

# Detect known faces in the image
face_locations, face_names = sfr.detect_known_faces(frame)

# Draw rectangles and labels around detected faces
for face_loc, name in zip(face_locations, face_names):
    y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

    # Display the name above the face
    cv.putText(small_frame, name, (x1, y1 - 10), cv.FONT_HERSHEY_DUPLEX, 1, (170, 170, 200), 1)
    
    # Draw a rectangle around the face
    cv.rectangle(small_frame, (x1, y1), (x2, y2), (170, 170, 170), 1)

# Display the frame with face recognition results
cv.imshow("Image with Face Recognition", small_frame)
cv.waitKey(0)

# Close all windows
cv.destroyAllWindows()
