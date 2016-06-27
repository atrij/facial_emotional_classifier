#Imports
from BagOfVisualWords import BagOfVisualWords
from DatasetOrganiser import DatasetOrganiser
from ImageProcessor import ImageProcessor
from ImageProvider import ImageProvider
from PrincipalComponentAnalysis import PrincipalComponentAnalysis
from SIFT import SIFT

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
datasetPathEmotions = "source_emotions"
datasetPathImages = "source_images"

# Organise dataset
DatasetOrganiser.organiseDataset(emotions, datasetPathEmotions, datasetPathImages)

# Pre-process the images and save
for emotion in emotions:
    ImageProcessor.performPreprocessing(emotion)

# Split into training data and test data
trainingData, testData = ImageProvider.splitDataset(emotions)

# Eg. of a training data
print trainingData["anger"][0]

# Eg. of a test data
print testData["anger"][0]

#Create imageDictionary for training Data
imageDictionary = ImageProvider.getImageDictionaryFromFilePaths(trainingData, emotions)

# Eg. of an image
print imageDictionary["anger"][0].shape # 3 Dimensional vector

# Create a SIFT object
sift = SIFT(0, 3, 0.03, 10, 1.6)

#Detect keypoints
keyPointsDictionary = sift.detectKeyPointsFromImageDictionary(imageDictionary, emotions)

# Eg. of a keypoint
print keyPointsDictionary["anger"]

# Get descriptors
descriptorsDictionary = sift.computeDescriptors(imageDictionary, keyPointsDictionary, emotions)

print len(descriptorsDictionary["anger"])

# Applying PCA - Principal Component Analysis - It is used for dimensionality reduction
eigenvectorsDictionary = PrincipalComponentAnalysis.computeEigenvectors(descriptorsDictionary, emotions)

#The next step is to use Bag of visual words approach.
# Create Training Data for clustering
trainingDescriptorsList = []
clusterCount = 100

for emotion in emotions:

    eigenvectorsList = eigenvectorsDictionary[emotion]

    for p in eigenvectorsList:
        for q in p:
            trainingDescriptorsList.append(q)

bagOfVisualWords = BagOfVisualWords(clusterCount, trainingDescriptorsList)
bagOfVisualWords.getHistogramForImages(eigenvectorsDictionary, emotions)


