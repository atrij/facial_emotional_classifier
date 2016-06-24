import cv2

class SIFT:

    sift = None;

    def __init__(self, features, octaveLayers, contrastThreshold, edgeThreshold, sigma):
        self.features = features
        self.octaveLayers = octaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma
        self.sift = cv2.xfeatures2d.SIFT_create(self.features, self.octaveLayers, self.contrastThreshold, self.edgeThreshold,
                                           self.sigma)

    def detectKeyPoints(self, image):
        return self.sift.detect(image, None)

    def computeDescriptors(self, image, keypoints):
        kp, des = self.sift.compute(image, keypoints)
        return des
