import cv2

class ImageProcessor:

    @staticmethod
    def convertImgToGrayScale(image):
        grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print grayScaleImage.shape
        return grayScaleImage

    @staticmethod
    def convertImgListToGrayScale(imageList):
        grayScaleImageList = []
        for image in imageList:
            grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grayScaleImageList.append(grayScaleImage)
            print grayScaleImage.shape

        return grayScaleImageList