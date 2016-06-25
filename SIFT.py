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

    def detectKeyPointsFromImageList(self, imageList):
        keypointsList = []

        for image in imageList:
            keypointsList.append(self.sift.detect(image, None))

        return keypointsList

    def computeDescriptors(self, imageList, keypointsList):
        length = min(len(imageList), len(keypointsList))
        i = 0
        descriptorList = []

        while(i<length):
            kp, des = self.sift.compute(imageList[i], keypointsList[i])
            descriptorList.append(des)
            i = i+1

        return descriptorList
