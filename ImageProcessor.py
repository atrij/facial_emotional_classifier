import cv2
from ImageProvider import ImageProvider


class ImageProcessor:

    @staticmethod
    def convertImgToGrayScale(image):
        grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayScaleImage

    @staticmethod
    def convertImgListToGrayScale(imageList):
        grayScaleImageList = []
        for image in imageList:
            grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grayScaleImageList.append(grayScaleImage)

        return grayScaleImageList


    @staticmethod
    def performPreprocessing(emotion):
        files = ImageProvider.getImages(emotion)

        fileNumber = 0

        for file in files:
            image = cv2.imread(file)
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("dataset\\%s\\%s.jpg" % (emotion, fileNumber), grayImage)
            fileNumber = fileNumber + 1
