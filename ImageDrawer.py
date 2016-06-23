import cv2

class ImageDrawer:

    def drawKeypoints(self, image, keypoints):
        image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('sift_keypoints1.jpg', image)