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
        printCounter = 0

        for emotion in emotions:
            keypointsList = []
            imageList = imageDictionary[emotion]

            for image in imageList:

                if(printCounter == 0):
                    print ("Example of a keypoint - ", self.sift.detect(image, None))
                    printCounter = 1

                keypointsList.append(self.sift.detect(image, None))

            keyPointDictionary[emotion] = keypointsList

        print ("Keypoint Dictionary - ", keyPointDictionary)
        print ("Keypoint List for one emotion - ", keyPointDictionary["anger"])
        print ("One keypoint for anger - ", keyPointDictionary["anger"][0])
        return keyPointDictionary

    def computeDescriptors(self, imageDictionary, keypointsDictionary, emotions):

        descriptorsDictionary = {}
        printCounter = 0

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
                if(printCounter ==0):
                    print ("Example of a descriptor - ", des)
                    print ("Shape of the descriptor - ", des.shape)
                    printCounter = 1

            descriptorsDictionary[emotion] = descriptorList

        print ("Descriptor Dictionary - ", descriptorsDictionary)
        print ("Descriptor List for one emotion - ", descriptorsDictionary["anger"])
        print ("One descriptor for anger - ", descriptorsDictionary["anger"][0])
        return descriptorsDictionary
