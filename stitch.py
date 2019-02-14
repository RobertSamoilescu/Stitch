import cv2
import imutils
import numpy as np


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def color_correction(self, result, image, mask, stitch="Right"):
        # color difference
        if stitch == "Right":
            rows, cols = np.where(mask[0:image.shape[0], 0:image.shape[1]] > 0)
        else:
            rows, cols = np.where(mask[0:image.shape[0], image.shape[1]//2:] > 0)
            cols += image.shape[1]//2

        diff = (image[rows, cols] - result[rows, cols]).mean(axis=0)
        result[mask > 0] += diff.reshape(1, -1)
        result = np.clip(result, 0, 255.0)

        return result 


    def blend(self, result, image, stitch="Right"):
        mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = result.astype(dtype=np.float32)
        image = image.astype(dtype=np.float32)

        # color correction
        result = self.color_correction(result, image, mask, stitch)
       
        # blend edge
        for i in range(result.shape[0]):
            if stitch == "Right":
                cols, = np.where(mask[i, 0:image.shape[1]] > 0) 
            else:
                cols, = np.where(mask[i, image.shape[1]//2:] > 0)
                cols += image.shape[1] // 2

            if len(cols) == 0:
                continue

            inf, sup = cols.min(), cols.max()
            # Maybe try different blending functions or different limits for sigmoid
            # factors = np.linspace(0, 1, sup - inf + 1).reshape(-1, 1)
            factors = self.sigmoid(np.linspace(-10, 10, sup - inf + 1)).reshape(-1, 1)

            if stitch != "Right":
                factors = 1 - factors

            result[i, inf:sup+1] *= factors
            image[i, inf:sup+1] *= 1 - factors

            if stitch == "Right":
                result[i, 0:image.shape[1]] += image[i, :]
            else:
                result[i, image.shape[1]//2:] += image[i, image.shape[1]//2:]

        return result.astype(dtype=np.uint8)


    def stitch(self, images,  H=np.array([]), stitch="Left", ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images

        if stitch == "Left":
            new_imageA = np.zeros((imageA.shape[0], 2 * imageA.shape[1], imageA.shape[2])).astype(np.uint8)
            new_imageA[0:imageA.shape[0], imageA.shape[1]:] = imageA
            imageA = new_imageA

        kps_computed = False
        if H.size == 0:
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)
            kps_computed = True

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
            result = self.blend(result, imageB, stitch)
        else:
            result = cv2.warpPerspective(imageB, H,
                                         (imageA.shape[1], imageB.shape[0]))
            result = self.blend(result, imageA, stitch)


        # check to see if the keypoint matches should be visualized
        if showMatches and kps_computed:
            if stitch == "Right":
                vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            else:
                vis = self.drawMatches(imageB, imageA, kpsB, kpsA, matches, status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis), H

        # return the stitched image
        return result, H


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
    def __init__(self):
        self.stitcher = Stitcher()
        self.leftH = np.array([])
        self.rightH = np.array([])

    def preprocess(self, leftImg, centerImg, rightImg, K=np.array([]), distCoeffs=np.array([])):
        leftImg = imutils.resize(leftImg, width=400)
        centerImg = imutils.resize(centerImg, width=400)
        rightImg = imutils.resize(rightImg, width=400)

        # undistor if K and dist
        if K.size != 0 and distCoeffs.size != 0:
            leftImg = cv2.undistort(leftImg, K, distCoeffs)
            centerImg = cv2.undistort(centerImg, K, distCoeffs)
            rightImg = cv2.undistort(rightImg, K, distCoeffs)

        return leftImg, centerImg, rightImg
       

    def get_stitched(self, leftImg, centerImg, rightImg, K=np.array([]), distCoeffs=np.array([])):
        leftImg, centerImg, rightImg = self.preprocess(leftImg, centerImg, rightImg, K, distCoeffs)
        
        # stitch images
        if self.leftH.size == 0 or self.rightH.size == 0:
            result, self.leftH = self.stitcher.stitch([leftImg, centerImg], stitch="Left")
            result, self.rightH = self.stitcher.stitch([result, rightImg], stitch="Right")
        else:
            result, _ = self.stitcher.stitch([leftImg, centerImg], self.leftH, stitch="Left")
            result, _ = self.stitcher.stitch([result, rightImg], self.rightH, stitch="Right")

        return result


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
    (result, vis), _ = stitcher.stitch([imageA, imageB], stitch="Left", showMatches=True)
    (result, vis), _ = stitcher.stitch([result, imageC], stitch="Right", showMatches=True)


    # show the images
    cv2.imshow("vis", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
