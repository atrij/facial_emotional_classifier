from domain.service.config.Constants import Constants


class Config:

    #Dataset
    isDatasetOrganised = False

    #Preprocessing
    isPreProcessingDone = False

    bovwClusterCount = 256

    @staticmethod
    def getDatasetName():
        return Constants.cohn_Kanade_extended

    @staticmethod
    def getDescriptorExractorMethodName():
        return Constants.sift

    @staticmethod
    def getDimensionalityReductionMethodName():
        return Constants.pca

    @staticmethod
    def getDescriptorPoolingMethodName():
        return Constants.bagOfVisualWords

    @staticmethod
    def getPreprocessingMethodList():
        preProcessingMethodList = [Constants.grayScaleConversion, Constants.faceDetectionHAAR]
        return preProcessingMethodList