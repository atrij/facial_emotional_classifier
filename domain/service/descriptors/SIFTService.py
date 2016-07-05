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
                    print ("Number of keypoints in a random image - ", len(self.sift.detect(image, None)))
                    printCounter = 1

                keypointsList.append(self.sift.detect(image, None))

            keyPointDictionary[emotion] = keypointsList

        print ("Length of Keypoint List for one emotion - ", len(keyPointDictionary["contempt"]))
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
                    print ("Number of descriptors for an image - ", len(des))
                    print ("Shape of the descriptor - ", des.shape)
                    printCounter = 1

            descriptorsDictionary[emotion] = descriptorList

        print ("Length of Descriptor List for one emotion - ", len(descriptorsDictionary["contempt"]))
        print ("One descriptor for contempt - ", descriptorsDictionary["contempt"][0])
        return descriptorsDictionary
