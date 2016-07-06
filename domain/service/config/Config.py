from domain.service.config.Constants import Constants


class Config:

    #Dataset
    isDatasetOrganised = False

    #Preprocessing
    isPreProcessingDone = False

    shouldPerformDimensionalityReduction = True

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
        preProcessingMethodList = [Constants.grayScaleConversion, Constants.faceDetectionHAAR, Constants.averaging]
        return preProcessingMethodList

    @staticmethod
    def checkConfigurations():

        if Config.isDatasetOrganised is False & Config.isPreProcessingDone is True:
            raise ValueError(" Preprocessing is required if dataset organisation is performed")

        if Config.getDatasetName() not in Constants.datasetList:
            raise ValueError("Invalid Dataset Name")

        if Config.getDescriptorExractorMethodName() not in Constants.desriptorExtractorMethodList:
            raise ValueError("Invalid DescriptorExtraction Method Name")

        if Config.getDescriptorPoolingMethodName() not in Constants.descriptorPoolingMethodList:
            raise ValueError("Invalid DescriptorPooling Method Name")

        if Config.shouldPerformDimensionalityReduction is True:
            if Config.getDimensionalityReductionMethodName() not in Constants.dimensionalityReductionMethodList:
                raise ValueError("Invalid DimensionalityReduction Method Name")
