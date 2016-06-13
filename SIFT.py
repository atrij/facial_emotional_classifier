#Imports
import cv2
import numpy as np

# Read an image. TEst it with a relevant image eg. smiling girl
img = cv2.imread("smiling-girl.jpg")

#COnvert image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.03, 10, 1.6);

#Detect keypoints
kp = sift.detect(gray, None)

#Draw keypoints
img=cv2.drawKeypoints(gray,kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)

# Get descriptors
kp, des = sift.compute(gray, kp)

print (len(kp))
print (des[0])