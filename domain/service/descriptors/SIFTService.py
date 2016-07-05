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
            keypointListList = []
            imageList = imageDictionary[emotion]

            if(emotion == "contempt"):
                print ("Length of imageList - ", len(imageList))

            for image in imageList:

                if(printCounter == 0):
                    print ("Number of keypoints in a random image - ", len(self.sift.detect(image, None)))
                    printCounter = 1

                keypointListList.append(self.sift.detect(image, None)) # Here keypointListList is a list of list.

            keyPointDictionary[emotion] = keypointListList

        print ("Length of KeypointListList for one emotion - ", len(keyPointDictionary["contempt"]))
        print ("Length of KeypointList for a random image in one emotion - ", len(keyPointDictionary["contempt"][0]))
        return keyPointDictionary

    def computeDescriptors(self, imageDictionary, keypointsDictionary, emotions):

        descriptorsDictionary = {}
        printCounter = 0

        for emotion in emotions:
            imageList = imageDictionary[emotion]
            keypointsList = keypointsDictionary[emotion]

            length = min(len(imageList), len(keypointsList))
            i = 0
            descriptorListList = []

            while(i<length):
                kp, des = self.sift.compute(imageList[i], keypointsList[i])
                descriptorListList.append(des)
                i = i+1
                if(printCounter ==0):
                    print ("Shape of the descriptor for the random image- ", des.shape)
                    printCounter = 1

            descriptorsDictionary[emotion] = descriptorListList

        print ("Length of DescriptorListList for one emotion - ", len(descriptorsDictionary["contempt"]))
        print ("Length of DescriptorList for a random image in one emotion - ", len(descriptorsDictionary["contempt"][0]))
        return descriptorsDictionary
