#Imports
from config.Constants import Constants
from dataset.DatasetOrganiser import DatasetOrganiser
from descriptors.DescriptorExtractor import DescriptorExtractor
from dimensionality_reduction.DimensionalityReducer import DimensionalityReducer
from image_operations.ImageProcessor import ImageProcessor
from image_operations.ImageProvider import ImageProvider
from pooling.DescriptorPooler import DescriptorPooler

emotions = Constants.emotions
datasetPathEmotions = Constants.datasetPathEmotions
datasetPathImages = Constants.datasetPathImages

# Organise dataset
DatasetOrganiser.organiseDataset(emotions, datasetPathEmotions, datasetPathImages)

# Pre-process the images and save
for emotion in emotions:
    ImageProcessor.performPreprocessing(emotion)

# Split into training data and test data
trainingData, testData = ImageProvider.splitDataset(emotions)

# Eg. of a training data
print trainingData[emotions[1]][0]

# Eg. of a test data
print testData[emotions[1]][0]

#Create imageDictionary for training Data
imageDictionary = ImageProvider.getImageDictionaryFromFilePaths(trainingData, emotions)

# Eg. of an image
print imageDictionary
print imageDictionary[emotions[1]][0].shape # 3 Dimensional vector

# Calculate Descriptors
argumentList = [0, 3, 0.03, 10, 1.6]
descriptorsDictionary = DescriptorExtractor.extractDescriptors("SIFT", argumentList, imageDictionary, emotions)

# Applying PCA - Principal Component Analysis - It is used for dimensionality reduction
pcaArgumentList = [descriptorsDictionary, emotions]
eigenvectorsDictionary = DimensionalityReducer.reduceDimensionality("PCA", pcaArgumentList)

# Eg. of an image's descriptors
print eigenvectorsDictionary[emotions[1]][0].shape # 128*128 matrix.

#The next step is to use Bag of visual words approach.

# Create Training Data for clustering
trainingDescriptorsList = []
clusterCount = 256

for emotion in emotions:

    eigenvectorsList = eigenvectorsDictionary[emotion]

    for p in eigenvectorsList:
        for q in p:
            trainingDescriptorsList.append(q)

print len(trainingDescriptorsList)

bovwArgumentList = [clusterCount, trainingDescriptorsList, eigenvectorsDictionary, emotions]
histogramDictionary = DescriptorPooler.poolDescriptors("BoVW", bovwArgumentList)


