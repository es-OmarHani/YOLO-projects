import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFaceRecognition:
    def __init__(self):
        self.known_face_encodings = []  # List to store face encodings
        self.known_face_names = []      # List to store corresponding face names

        # Resize frame for faster processing
        self.frame_resizing = 0.5

    def load_encoding_images(self, images_path):
        """
        Load encoding images from the specified path.
        
        Args:
            images_path (str): Path to the directory containing face images.
            
        Returns:
            None
        """
        # Load images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encodings and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            
            # Get face encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store filename and face encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        """
        Detect known faces in the given frame.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format.
            
        Returns:
            tuple: A tuple containing face locations (numpy array) and corresponding face names.
        """
        # Resize the frame
        width = int(frame.shape[1] * self.frame_resizing)
        height = int(frame.shape[0] * self.frame_resizing)
        dim = (width, height)
        small_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare face with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Adjust coordinates with frame resizing
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        
        return face_locations.astype(int), face_names
