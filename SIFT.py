#Imports
import cv2
import numpy as np

# Read an image. TEst it with a relevant image eg. smiling girl
img = cv2.imread("smiling-girl.jpg")
print ("Size of image - ", img.shape) # Color image is a 3 dimensional matrix with 334 rows and 500 columns

#COnvert image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print ('Size of gray image - ', gray.shape)  # Gray image is a 2 dimensional matrix with 334 rows and 500 columns

# Create a SIFT object
sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.03, 10, 1.6);

#Detect keypoints
kp = sift.detect(gray, None)
print ("Size of keypoints - ", len(kp)) #Keypoint is a 1 dimensional list consisting of 439 keypoints

#Draw keypoints
img=cv2.drawKeypoints(gray,kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)

# Get descriptors
kp, des = sift.compute(gray, kp)
print ("Size of descriptors - ", des.shape) #Descriptor is a n*128 matrix where n is number of keypoints. Thus corresponding to each keypoint, we get a vector with 128 values as a descriptor
print ("Length of first descriptor - ", len(des[0]))

