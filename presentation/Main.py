#Imports
from domain.service.config.Config import Config
from domain.service.config.Constants import Constants
from domain.service.dataset.DatasetService import DatasetService
from domain.service.descriptors.DescriptorExtractorService import DescriptorExtractor
from domain.service.descriptors_pooling.DescriptorPoolerService import DescriptorPooler
from domain.service.dimensionality_reduction.DimensionalityReducerService import DimensionalityReducerService
from domain.service.image_operations.ImagePreProcessService import ImagePreProcessService


def constructTrainingDescriptorsList():
    trainingDescriptorsList = []
    for emotion in emotions:

        eigenvectorListList = eigenvectorsDictionary[emotion]

        for p in eigenvectorListList:
            for q in p:
                trainingDescriptorsList.append(q)

    return trainingDescriptorsList

print " ----- START ------ "

# Check Configurations
if Config.isDatasetOrganised is False & Config.isPreProcessingDone is True:
    raise ValueError(" Preprocessing is required if dataset organisation is performed")

if Config.getDatasetName() not in Constants.datasetList:
    raise ValueError("Invalid Dataset Name")

if Config.getDescriptorExractorMethodName() not in Constants.desriptorExtractorMethodList:
    raise ValueError("Invalid DescriptorExtraction Method Name")

if Config.getDimensionalityReductionMethodName() not in Constants.dimensionalityReductionMethodList:
    raise ValueError("Invalid DimensionalityReduction Method Name")

if Config.getDescriptorPoolingMethodName() not in Constants.descriptorPoolingMethodList:
    raise ValueError("Invalid DescriptorPooling Method Name")

emotions = Constants.emotions
datasetPathEmotions = Constants.datasetPathEmotions
datasetPathImages = Constants.datasetPathImages

# Organise dataset
if Config.isDatasetOrganised is False:
    DatasetService.organiseDataset(Config.getDatasetName(), emotions, datasetPathEmotions, datasetPathImages)
    print "Dataset Organised"

# Pre-process the images and save
if Config.isPreProcessingDone is False:
    for emotion in emotions:
        ImagePreProcessService.performPreprocessing(Config.getPreprocessingMethodList(), emotion)
    print "Preprocessing step complete"

# Split into training data and test data
trainingData, testData = DatasetService.splitDataset(emotions)

# Eg. of a training data
print "Training File example - " + trainingData[emotions[2]][0]

# Eg. of a test data
print "Test file example - " + testData[emotions[2]][0]

#Create imageDictionary for training Data
imageDictionary = DatasetService.getImageDictionaryFromFilePaths(trainingData, emotions)

# Eg. of an image dictionary and image
print ("Shape of a random image -- ", imageDictionary[emotions[1]][0].shape) # 3 Dimensional vector

# Calculate Descriptors
siftArgumentList = [0, 3, 0.03, 10, 1.6]
descriptorsDictionary = DescriptorExtractor.extractDescriptors(Config.getDescriptorExractorMethodName(), siftArgumentList, imageDictionary, emotions)

if Config.shouldPerformPCA is True:
    # Applying PCA - Principal Component Analysis - It is used for dimensionality reduction
    pcaArgumentList = [descriptorsDictionary, emotions]
    eigenvectorsDictionary = DimensionalityReducerService.reduceDimensionality(Config.getDimensionalityReductionMethodName(), pcaArgumentList)
else:
    eigenvectorsDictionary = descriptorsDictionary

#The next step is to use Bag of visual words approach.
# Create Training Data for clustering
trainingDescriptorsList = constructTrainingDescriptorsList()
print ("Length of training Descriptors - ", len(trainingDescriptorsList))

bovwArgumentList = [Config.bovwClusterCount, trainingDescriptorsList, eigenvectorsDictionary, emotions]
histogramDictionary = DescriptorPooler.poolDescriptors(Config.getDescriptorPoolingMethodName(), bovwArgumentList)
