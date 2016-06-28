#Imports
from BagOfVisualWords import BagOfVisualWords
from Constants import Constants
from DatasetOrganiser import DatasetOrganiser
from ImageProcessor import ImageProcessor
from ImageProvider import ImageProvider
from PrincipalComponentAnalysis import PrincipalComponentAnalysis
from SIFT import SIFT

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
print imageDictionary[emotions[1]][0].shape # 3 Dimensional vector

# Create a SIFT object
sift = SIFT(0, 3, 0.03, 10, 1.6)

#Detect keypoints
keyPointsDictionary = sift.detectKeyPointsFromImageDictionary(imageDictionary, emotions)

# Eg. of an image's keypoints
print len(keyPointsDictionary[emotions[1]][0])

# Get descriptors
descriptorsDictionary = sift.computeDescriptors(imageDictionary, keyPointsDictionary, emotions)

# Eg. of an image's descriptors
print descriptorsDictionary[emotions[1]][0].shape # n*128 matrix where n is number of keypoints. Each row is the descriptor for one keypoint.

# Applying PCA - Principal Component Analysis - It is used for dimensionality reduction
eigenvectorsDictionary = PrincipalComponentAnalysis.computeEigenvectors(descriptorsDictionary, emotions)

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

bagOfVisualWords = BagOfVisualWords(clusterCount, trainingDescriptorsList)
histogramDictionary = bagOfVisualWords.getHistogramForImages(eigenvectorsDictionary, emotions)


