import cv2

class ImageDrawerService:

    @staticmethod
    def drawKeypoints(imageList, keypointList):
        length = min(len(imageList), len(keypointList))
        i = 0

        while(i<length):
            image = cv2.drawKeypoints(imageList[i], keypointList[i], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(str(i) + ' - sift_keypoints.jpg', image)
            i = i+1

