from SIFTService import SIFT
from domain.service.config.Constants import Constants


class DescriptorExtractor:
    @staticmethod
    def extractDescriptors(modelName, argumentList, imageDictionary, emotions):
        if (modelName == Constants.sift):
            sift = SIFT(argumentList[0], argumentList[1], argumentList[2], argumentList[3], argumentList[4])

            # Detect keypoints
            keyPointsDictionary = sift.detectKeyPointsFromImageDictionary(imageDictionary, emotions)

            # Get descriptors
            descriptorsDictionary = sift.computeDescriptors(imageDictionary, keyPointsDictionary, emotions)

            return descriptorsDictionary