import cv2

class SIFT:

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

    def detectKeyPointsFromImageDictionary(self, imageDictionary, emotions):

        keyPointDictionary = {}

        for emotion in emotions:
            keypointsList = []
            imageList = imageDictionary[emotion]

            for image in imageList:
                keypointsList.append(self.sift.detect(image, None))

            keyPointDictionary[emotion] = keypointsList

        return keyPointDictionary

    def computeDescriptors(self, imageDictionary, keypointsDictionary, emotions):

        descriptorsDictionary = {}

        for emotion in emotions:
            imageList = imageDictionary[emotion]
            keypointsList = keypointsDictionary[emotion]

            length = min(len(imageList), len(keypointsList))
            i = 0
            descriptorList = []

            while(i<length):
                kp, des = self.sift.compute(imageList[i], keypointsList[i])
                descriptorList.append(des)
                i = i+1

            descriptorsDictionary[emotion] = descriptorList

        return descriptorsDictionary
