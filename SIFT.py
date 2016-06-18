#Imports
import cv2
import numpy as np

#This is just a test class.

# Read the images. TEst it with a relevant image eg. smiling girl
img1 = cv2.imread("smiling-girl.jpg")
print ("Shape of image1 - ", img1.shape) # Color image is a 3 dimensional matrix

img2 = cv2.imread("smiling_boy.jpg")
print ("Shape of image2 - ", img2.shape)

img3 = cv2.imread("crying_boy.jpg")
print ("Shape of image3 - ", img3.shape)

#Convert images to gray scale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
print ('Shape of gray image1 - ', gray1.shape)  # Gray image is a 2 dimensional matrix

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
print ('Shape of gray image2 - ', gray2.shape)

gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
print ('Shape of gray image3 - ', gray3.shape)

# Create a SIFT object
sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.03, 10, 1.6);

#Detect keypoints
kp1 = sift.detect(gray1, None)
print ("Length of keypoints1 - ", len(kp1)) #Keypoint is a 1 dimensional list consisting of 439 keypoints

kp2 = sift.detect(gray2, None)
print ("Length of keypoints2 - ", len(kp2)) #Keypoint is a 1 dimensional list consisting of 138 keypoints

kp3 = sift.detect(gray3, None)
print ("Length of keypoints3 - ", len(kp3)) #Keypoint is a 1 dimensional list consisting of 908 keypoints

#Draw keypoints
img1=cv2.drawKeypoints(gray1,kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints1.jpg',img1)

img2=cv2.drawKeypoints(gray2,kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints2.jpg',img2)

img3=cv2.drawKeypoints(gray3,kp3, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints3.jpg',img3)

# Get descriptors
kp1, des1 = sift.compute(gray1, kp1)
print ("Shape of descriptors1 - ", des1.shape) #Descriptor is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor

kp2, des2 = sift.compute(gray2, kp2)
print ("Shape of descriptors2 - ", des2.shape) #Descriptor is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor

kp3, des3 = sift.compute(gray3, kp3)
print ("Shape of descriptors3 - ", des3.shape) #Descriptor is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor

# Applying PCA - Principal Component Analysis - It is used for dimensionality reduction
mean1, eigenvectors1 = cv2.PCACompute(des1, mean = np.array([]))
mean2, eigenvectors2 = cv2.PCACompute(des2, mean = np.array([]))
mean3, eigenvectors3 = cv2.PCACompute(des3, mean = np.array([]))

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

