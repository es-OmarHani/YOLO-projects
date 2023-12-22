import cv2
import matplotlib.pyplot as plt
import numpy as np

from util import get_parking_spots_bboxes, empty_or_not

# Paths for the mask and video
mask_path = 'images/mask_1920_1080.png'
video_path = 'videos/parking_1920_1080.mp4'

# Read the mask image in grayscale
mask = cv2.imread(mask_path, 0)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Connected components analysis on the mask
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
# print(f"length of connected_components: {connected_components[0]}")

# Extract parking spot bounding boxes from connected components
spots = get_parking_spots_bboxes(connected_components)

# Initialize lists to store spot status and differences
spots_status = [None for _ in spots]
diffs = [None for _ in spots]

# Initialize variables for frame processing
previous_frame = None
frame_nmr = 0
ret = True
step = 30


# Function to calculate the absolute difference between the means of two images
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# Loop through video frames
while ret:
    ret, frame = cap.read()

    # Process frames every 'step' frames which means every one second (30fps)
    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1 : y1+h, x1 : x1+w, :]
            
            # Calculate and store the difference for each spot
            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1 : y1+h, x1 : x1+w, :])

        # Print the differences sorted in descending order
        # print([diffs[j] for j in np.argsort(diffs)][::-1])

        # # plot hist to know region of threshold for movement in spot between 2 frames
        # plt.hist([diffs[j] / np.amax(diffs) for j in np.argsort(diffs)][::-1], bins = 20)
        # plt.show()

    if frame_nmr % step == 0:
        # Update spot status based on differences
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
            print(f"arr : {arr_} | len: {len(arr_)}")

            # for j in np.argsort(diffs) :
            #     if diffs[j] / np.amax(diffs) > 0.4 :
            #         print(f"val is : {j[-1]}")

        for spot_indx in arr_:

            spot = spots[spot_indx]
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            # Determine and store the spot status
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status

    if frame_nmr % step == 0:
        # Update the previous frame
        previous_frame = frame.copy()

    # Draw rectangles on the frame based on spot status
    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        color = (0, 255, 0) if spot_status else (0, 0, 255)
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    # Display information about available spots
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))),
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

