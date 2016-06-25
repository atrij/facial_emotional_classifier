import cv2

class ImageProcessor:

    @staticmethod
    def convertImgToGrayScale(image):
        grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print grayScaleImage.shape
        return grayScaleImage