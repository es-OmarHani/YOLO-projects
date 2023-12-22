import pickle
from skimage.transform import resize
import numpy as np
import cv2

# Constants for better readability
EMPTY = True
NOT_EMPTY = False

# Load the pre-trained model
MODEL = pickle.load(open("models/model.p", "rb"))

def empty_or_not(spot_bgr):
    """
    Determines if a parking spot is empty or not based on the input image.

    Args:
    - spot_bgr (numpy.ndarray): The input image of a parking spot in BGR format.

    Returns:
    - bool: True if the parking spot is empty, False otherwise.
    """
    flat_data = []

    # Resize the input image
    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    # Predict using the pre-trained model
    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY

def get_parking_spots_bboxes(connected_components):
    """
    Extracts bounding boxes of parking spots from connected components.

    Args:
    - connected_components (tuple): Result of connected components analysis.

    Returns:
    - list: A list of bounding boxes for parking spots in the format [x, y, width, height].
    """
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1  # Scaling coefficient, adjust as needed

    for i in range(1, totalLabels):
        # Extract coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots
