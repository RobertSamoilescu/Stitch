import cv2
import imutils
import numpy as np


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def stitch(self, images,  stitch="Left", ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images

        if stitch == "Left":
            new_imageA = np.zeros((imageA.shape[0], imageA.shape[1] + imageB.shape[1], imageA.shape[2])).astype(np.uint8)
            new_imageA[0:imageA.shape[0], imageA.shape[1]:] = imageA
            imageA = new_imageA

        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        if stitch == "Right":
            M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        else:
            M = self.matchKeypoints(kpsB, kpsA, featuresB, featuresA, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M

        if stitch == "Right":
            result = cv2.warpPerspective(imageA, H,
                                         (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
            result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        else:
            result = cv2.warpPerspective(imageB, H,
                                         (imageA.shape[1], imageB.shape[0]))
            result[0:imageA.shape[0], imageB.shape[1]:] = imageA[0:imageA.shape[0], imageB.shape[1]:]


        # check to see if the keypoint matches should be visualized
        if showMatches:
            if stitch == "Right":
                vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            else:
                vis = self.drawMatches(imageB, imageA, kpsB, kpsA, matches, status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result


    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)


    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None


    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis




class Stitcher3(object):
    def __init__(self, leftImg, centerImg, rightImg, K=np.array([]), distCoeffs=np.array([])):
        self.leftImg = leftImg
        self.centerImg = centerImg
        self.rightImg = rightImg


        self.leftImg = imutils.resize(self.leftImg, width=400)
        self.centerImg = imutils.resize(self.centerImg, width=400)
        self.rightImg = imutils.resize(self.rightImg, width=400)

        # undistor if K and dist
        if K.size != 0 and distCoeffs.size != 0:
            self.K = K
            self.distCoeffs = distCoeffs

            self.leftImg = cv2.undistort(self.leftImg, K, distCoeffs)
            self.centerImg = cv2.undistort(self.centerImg, K, distCoeffs)
            self.rightImg = cv2.undistort(self.rightImg, K, distCoeffs)

        # stitch images
        stitcher = Stitcher()
        self.result = stitcher.stitch([self.leftImg, self.centerImg], stitch="Left")
        self.result = stitcher.stitch([self.result, self.rightImg], stitch="Right")


    def get_stitched(self):
        return self.result


if __name__ == "__main__":
    # camera params
    dist = np.array([0.053314, -0.117603, -0.004064, -0.001819, 0.000000])

    K = np.array([
            [1173.122620, 0.000000, 969.335924],
            [0.000000, 1179.612539, 549.524382],
            [0.000000,  0.000000, 1.000000]
        ])


    """
    imageA = cv2.imread("imgs/cam_1.png")
    imageA = imutils.resize(imageA, width=400)
    imageA = cv2.undistort(imageA, K, dist)

    imageB = cv2.imread("imgs/cam_2.png")
    imageB = imutils.resize(imageB, width=400)
    imageB = cv2.undistort(imageB, K, dist)

    imageC = cv2.imread("imgs/cam_3.png")
    imageC = imutils.resize(imageC, width=400)
    imageC = cv2.undistort(imageC, K, dist)
    """


    imageA = cv2.imread("imgs/1.jpg")
    imageA = imutils.resize(imageA, width=400)

    imageB = cv2.imread("imgs/2.jpg")
    imageB = imutils.resize(imageB, width=400)

    imageC = cv2.imread("imgs/3.jpg")
    imageC = imutils.resize(imageC, width=400)


    # stitch the images together to create a panorama
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], stitch="Left", showMatches=True, )
    (result, vis) = stitcher.stitch([result, imageC], stitch="Right", showMatches=True)


    # show the images
    cv2.imshow("vis", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
