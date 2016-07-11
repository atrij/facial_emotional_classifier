from SIFTService import SIFTService
from domain.service.config.Constants import Constants


class DescriptorExtractorService:
    @staticmethod
    def extractDescriptors(modelName, argumentList, imageDictionary, emotions):
        if (modelName == Constants.sift):
            sift = SIFTService(argumentList[0], argumentList[1], argumentList[2], argumentList[3], argumentList[4])

            # Detect keypoints
            keyPointsDictionary = sift.detectKeyPointsFromImageDictionary(imageDictionary, emotions)

            # Get descriptors
            descriptorsDictionary = sift.computeDescriptors(imageDictionary, keyPointsDictionary, emotions)

            return descriptorsDictionary