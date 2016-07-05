from domain.service.descriptors_pooling.BagOfVisualWordsService import BagOfVisualWords


class DescriptorPooler:

    @staticmethod
    def poolDescriptors(modelName, argumentList):

        if(modelName == "BoVW"):
            bagOfVisualWords = BagOfVisualWords(argumentList[0], argumentList[1])
            histogramDictionary = bagOfVisualWords.getHistogramForImages(argumentList[2], argumentList[3])
            return histogramDictionary