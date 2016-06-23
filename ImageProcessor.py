import cv2

class ImageProcessor:

    def convertImgToGrayScale(self, image):
        grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print grayScaleImage.shape
        return grayScaleImage