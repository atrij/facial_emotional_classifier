#Imports
from BagOfVisualWords import BagOfVisualWords
from DatasetOrganiser import DatasetOrganiser
from ImageDrawer import ImageDrawer
from ImageProcessor import ImageProcessor
from ImageProvider import ImageProvider
from PrincipalComponentAnalysis import PCA
from SIFT import SIFT

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
datasetPathEmotions = "source_emotions"
datasetPathImages = "source_images"
# Organise dataset
DatasetOrganiser.organiseDataset(emotions, datasetPathEmotions, datasetPathImages)

# Read the images.
imgList = ImageProvider.getImages() # Should pass the database path as parameter

#Convert images to gray scale
grayScaleImageList = ImageProcessor.convertImgListToGrayScale(imgList)

# Some processing to be done on image for standardisation

# Create a SIFT object
sift = SIFT(0, 3, 0.03, 10, 1.6)

#Detect keypoints
keyPointsList = sift.detectKeyPointsFromImageList(grayScaleImageList)

print ("Length of keypoints1 - ", len(keyPointsList[0])) #KeypointList[0] is a 1 dimensional list consisting of 439 keypoints
print ("Length of keypoints2 - ", len(keyPointsList[1])) #KeypointList[1] is a 1 dimensional list consisting of 138 keypoints
print ("Length of keypoints3 - ", len(keyPointsList[2])) #KeypointList[2] is a 1 dimensional list consisting of 908 keypoints

#Draw keypoints
ImageDrawer.drawKeypoints(grayScaleImageList, keyPointsList)

# Get descriptors
descriptorsList = sift.computeDescriptors(grayScaleImageList, keyPointsList)
print ("Shape of descriptors1 - ", descriptorsList[0].shape) #DescriptorList[0] is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor
print ("Shape of descriptors2 - ", descriptorsList[1].shape) #DescriptorList[1] is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor
print ("Shape of descriptors3 - ", descriptorsList[2].shape) #DescriptorList[2] is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor

# Applying PCA - Principal Component Analysis - It is used for dimensionality reduction
eigenvectorsList = PCA.computeEigenvectors(descriptorsList)

print("Shape of eigenvector1 - ", eigenvectorsList[0].shape)
print("Shape of eigenvector2 - ", eigenvectorsList[1].shape)
print("Shape of eigenvector3 - ", eigenvectorsList[2].shape)

#The next step is to use Bag of visual words approach.
# Create Training Data for clustering
trainingDescriptorsList = []
clusterCount = 100

for p in eigenvectorsList:
    for q in p:
        trainingDescriptorsList.append(q)

bagOfVisualWords = BagOfVisualWords(clusterCount, trainingDescriptorsList)
bagOfVisualWords.getHistogramForImages(eigenvectorsList)


