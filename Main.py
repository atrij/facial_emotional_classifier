#Imports
import cv2
import numpy as np

from ImageDrawer import ImageDrawer
from ImageProcessor import ImageProcessor
from ImageProvider import ImageProvider
from PCA import PCA
from SIFT import SIFT

# Read the images.
imgList = ImageProvider.getImages() # Should pass the database path as parameter

#Convert images to gray scale
grayScaleImageList = []
for image in imgList:
    grayScaleImageList.append(ImageProcessor.convertImgToGrayScale(image))

# Some processing to be done on image for standardisation

# Create a SIFT object
sift = SIFT(0, 3, 0.03, 10, 1.6)

#Detect keypoints
keyPointsList = []
for grayScaleImage in grayScaleImageList:
    keyPointsList.append(sift.detectKeyPoints(grayScaleImage))

print ("Length of keypoints1 - ", len(keyPointsList[0])) #Keypoint is a 1 dimensional list consisting of 439 keypoints
print ("Length of keypoints2 - ", len(keyPointsList[1])) #Keypoint is a 1 dimensional list consisting of 138 keypoints
print ("Length of keypoints3 - ", len(keyPointsList[2])) #Keypoint is a 1 dimensional list consisting of 908 keypoints

#Draw keypoints
ImageDrawer.drawKeypoints(grayScaleImageList, keyPointsList)

# Get descriptors
descriptorsList = sift.computeDescriptors(grayScaleImageList, keyPointsList)
print ("Shape of descriptors1 - ", descriptorsList[0].shape) #Descriptor is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor
print ("Shape of descriptors2 - ", descriptorsList[1].shape) #Descriptor is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor
print ("Shape of descriptors3 - ", descriptorsList[2].shape) #Descriptor is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor

# Applying PCA - Principal Component Analysis - It is used for dimensionality reduction
eigenvectorsList = PCA.computeEigenvectors(descriptorsList)

print("Shape of eigenvector1 - ", eigenvectorsList[0].shape)
print("Shape of eigenvector2 - ", eigenvectorsList[1].shape)
print("Shape of eigenvector3 - ", eigenvectorsList[2].shape)

#The next step is to use Bag of visual words approach
bow = cv2.BOWKMeansTrainer(100) # Create a BoW object with 100 clusters

bow.add(eigenvectorsList[0])
bow.add(eigenvectorsList[1])
bow.add(eigenvectorsList[2])

dictionary = bow.cluster() # Creates a dictionary of visual words (Centroids of each of the clusters)
print ("Dictionary - ", dictionary.shape)


