import cv2
import numpy as np
import stitch
import imutils
import transformation

dist = np.array([0.053314, -0.117603, -0.004064, -0.001819, 0.000000])

K = np.array([
    [1173.122620, 0.000000, 969.335924],
    [0.000000, 1179.612539, 549.524382],
    [0.000000, 0.000000, 1.000000]
])

# # input videos
# leftCap = cv2.VideoCapture('video4/small_1.mp4')
# centerCap = cv2.VideoCapture("video4/small_3.mp4")
# rightCap = cv2.VideoCapture("video4/small_2.mp4")


# leftCap = cv2.VideoCapture('video2/small_3.mp4')
# centerCap = cv2.VideoCapture("video2/small_2.mp4")
# rightCap = cv2.VideoCapture("video2/small_1.mp4")

leftCap = cv2.VideoCapture('video1/small_3.mp4')
centerCap = cv2.VideoCapture("video1/small_1.mp4")
rightCap = cv2.VideoCapture("video1/small_2.mp4")

# Check if camera opened successfully
if (leftCap.isOpened() == False) or (centerCap.isOpened() == False) or (rightCap.isOpened() == False):
    print("Error opening video stream or file")



# Read until video is completed
sticher3 = stitch.Stitcher3()
while True:
    # Capture frame-by-frame
    retLeft, leftFrame = leftCap.read()
    retCenter, centerFrame = centerCap.read()
    retRight, rightFrame = rightCap.read()

    if retLeft and retCenter and retRight:
        leftFrame = transformation.Crop.crop_center(leftFrame, up=0.1, down=0.5, left=0.5, right=0.5)
        centerFrame = transformation.Crop.crop_center(centerFrame, up=0.1, down=0.5, left=0.5, right=0.5)
        rightFrame = transformation.Crop.crop_center(rightFrame, up=0.1, down=0.5, left=0.5, right=0.5)
        
        # stich the three images
        frame = sticher3.get_stitched(leftFrame, centerFrame, rightFrame)
       
        # Display the resulting frame
        frame = imutils.resize(frame, height=300)
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
leftCap.release()
centerCap.release()
rightCap.release()

# Closes all the frames
cv2.destroyAllWindows()
