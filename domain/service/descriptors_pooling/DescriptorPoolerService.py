from domain.service.config.Constants import Constants
from domain.service.descriptors_pooling.BagOfVisualWordsService import BagOfVisualWordsService


class DescriptorPoolerService:

    @staticmethod
    def poolDescriptors(modelName, argumentList):

        if(modelName == Constants.bagOfVisualWords):
            bagOfVisualWords = BagOfVisualWordsService(argumentList[0], argumentList[1])
            histogramDictionary = bagOfVisualWords.getHistogramForImages(argumentList[2], argumentList[3])
            return histogramDictionary