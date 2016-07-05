from SIFTService import SIFT


class DescriptorExtractor:
    @staticmethod
    def extractDescriptors(modelName, argumentList, imageDictionary, emotions):
        if (modelName == 'SIFT'):
            sift = SIFT(argumentList[0], argumentList[1], argumentList[2], argumentList[3], argumentList[4])

            # Detect keypoints
            keyPointsDictionary = sift.detectKeyPointsFromImageDictionary(imageDictionary, emotions)

            # Eg. of an image's keypoints
            print len(keyPointsDictionary[emotions[1]][0])

            # Get descriptors
            descriptorsDictionary = sift.computeDescriptors(imageDictionary, keyPointsDictionary, emotions)

            # Eg. of an image's descriptors
            print descriptorsDictionary[emotions[1]][0].shape  # n*128 matrix where n is number of keypoints. Each row is the descriptor for one keypoint.

            return descriptorsDictionary