#Imports
import cv2
import numpy as np

img = cv2.imread("table.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.03, 10, 1.6);
kp = sift.detect(gray, None)

img=cv2.drawKeypoints(gray,kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)

kp, des = sift.compute(gray, kp)

print (len(kp))
print (des[0])