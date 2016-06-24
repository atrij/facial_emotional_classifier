#Imports
import cv2
import numpy as np

from ImageDrawer import ImageDrawer
from ImageProcessor import ImageProcessor
from ImageProvider import ImageProvider
from PCA import PCA
from SIFT import SIFT

#This is just a test class.

# Read the images.
imageProvider = ImageProvider()
imgList = imageProvider.getImages()

#Convert images to gray scale
grayScaleImages = []
imageProcessor = ImageProcessor()
for image in imgList:
    grayScaleImages.append(imageProcessor.convertImgToGrayScale(image))

# Create a SIFT object
sift = SIFT(0, 3, 0.03, 10, 1.6)

#Detect keypoints
keyPoints = []
for grayScaleImage in grayScaleImages:
    keyPoints.append(sift.detectKeyPoints(grayScaleImage))

print ("Length of keypoints1 - ", len(keyPoints[0])) #Keypoint is a 1 dimensional list consisting of 439 keypoints
print ("Length of keypoints2 - ", len(keyPoints[1])) #Keypoint is a 1 dimensional list consisting of 138 keypoints
print ("Length of keypoints3 - ", len(keyPoints[2])) #Keypoint is a 1 dimensional list consisting of 908 keypoints

#Draw keypoints
imageDrawer = ImageDrawer()
imageDrawer.drawKeypoints(grayScaleImages[0], keyPoints[0])
imageDrawer.drawKeypoints(grayScaleImages[1], keyPoints[1])
imageDrawer.drawKeypoints(grayScaleImages[2], keyPoints[2])

# Get descriptors
descriptors1 = sift.computeDescriptors(grayScaleImages[0], keyPoints[0])
print ("Shape of descriptors1 - ", descriptors1.shape) #Descriptor is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor

descriptors2 = sift.computeDescriptors(grayScaleImages[1], keyPoints[1])
print ("Shape of descriptors2 - ", descriptors2.shape) #Descriptor is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor

descriptors3 = sift.computeDescriptors(grayScaleImages[2], keyPoints[2])
print ("Shape of descriptors3 - ", descriptors3.shape) #Descriptor is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor

# Applying PCA - Principal Component Analysis - It is used for dimensionality reduction
pca = PCA()
eigenvectors1 = pca.computeEigenvectors(descriptors1)
eigenvectors2 = pca.computeEigenvectors(descriptors2)
eigenvectors3 = pca.computeEigenvectors(descriptors3)

print("Shape of eigenvector1 - ", eigenvectors1.shape)
print("Shape of eigenvector2 - ", eigenvectors2.shape)
print("Shape of eigenvector3 - ", eigenvectors3.shape)

#The next step is to use Bag of visual words approach

bow = cv2.BOWKMeansTrainer(100) # Create a BoW object with 100 clusters

bow.add(eigenvectors1)
bow.add(eigenvectors2)
bow.add(eigenvectors3)

dictionary = bow.cluster() # Creates a dictionary of visual words (Centroids of each of the clusters)
print ("Dictionary - ", dictionary.shape)


