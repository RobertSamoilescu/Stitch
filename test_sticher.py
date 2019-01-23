import numpy as np
import cv2
import stitch
import imutils

if __name__ == "__main__":
    dist = np.array([0.053314, -0.117603, -0.004064, -0.001819, 0.000000])

    K = np.array([
        [1173.122620, 0.000000, 969.335924],
        [0.000000, 1179.612539, 549.524382],
        [0.000000, 0.000000, 1.000000]
    ])

    # read images
    leftImg = cv2.imread("imgs/cam_1.png")
    centerImg = cv2.imread("imgs/cam_2.png")
    rightImg = cv2.imread("imgs/cam_3.png")

    # create stitcher object
    stitcher3 = stitch.Stitcher3(leftImg, centerImg, rightImg)
    img = stitcher3.get_stitched()

    img = imutils.resize(img, width=1200)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
