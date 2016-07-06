import cv2

from domain.service.config.Constants import Constants
from domain.service.dataset.DatasetService import DatasetService


class ImagePreProcessService:
    faceDet1 = cv2.CascadeClassifier("/Users/harsh/opencv/data/haarcascades/haarcascade_frontalface_default.xml")
    faceDet2 = cv2.CascadeClassifier("/Users/harsh/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml")
    faceDet3 = cv2.CascadeClassifier("/Users/harsh/opencv/data/haarcascades/haarcascade_frontalface_alt.xml")
    faceDet4 = cv2.CascadeClassifier("/Users/harsh/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml")

    @staticmethod
    def performPreprocessing(methodList, emotion):

        grayScaleConversion = False
        faceDetectionHAAR = False
        averaging = False
        gaussianBlur = False
        medianBlur = False
        bilateralFiltering = False

        # Check if grayScale conversion is required
        if Constants.grayScaleConversion in methodList:
            grayScaleConversion = True

        # Check if face detection is required
        if Constants.faceDetectionHAAR in methodList:
            faceDetectionHAAR = True

        # Check if averaging is required
        if Constants.averaging in methodList:
            averaging = True

        # Check if gaussian Blur is required
        if Constants.gaussianBlur in methodList:
            gaussianBlur = True

        # Check if median blur is required
        if Constants.medianBlur in methodList:
            medianBlur = True

        # Check if bilateral filtering is required
        if Constants.bilateralFiltering in methodList:
            bilateralFiltering = True

        files = DatasetService.getImages(emotion)
        print ("Number of images before preprocessing for %s is %d", (emotion, len(files)))

        fileNumber = 0

        for file in files:
            image = cv2.imread(file)

            if grayScaleConversion is True:
                grayImage = ImagePreProcessService.__convertImageInGrayScale(image)

            outputImage = grayImage
            if faceDetectionHAAR is True:
                facefeatures = ImagePreProcessService.__detectFaceInImage(grayImage)
                outputImage = ImagePreProcessService.__cutFace(facefeatures, grayImage)

            averagedImage = outputImage
            if averaging is True:
                averagedImage = ImagePreProcessService.__applyImageAveraging(outputImage, 5)

            gaussianBlurredImage = averagedImage
            if gaussianBlur is True:
                gaussianBlurredImage = ImagePreProcessService.__applyGaussainBlur(averagedImage, 5, 0)

            medianBlurredImage = gaussianBlurredImage
            if medianBlur is True:
                medianBlurredImage = ImagePreProcessService.__applyMedianBlur(gaussianBlurredImage, 5)

            bilateralFilteredImage = medianBlurredImage
            if bilateralFiltering is True:
                bilateralFilteredImage = ImagePreProcessService.__applyBilateralFiltering(medianBlurredImage, 9, 75, 75)
                # Filter size: Large filters
                # (d > 5) are very slow, so it is recommended to use d=5 for real-time applications,
                # and perhaps d=9 for offline applications that need heavy noise filtering. Sigma values:
                # For simplicity, you can set the 2 sigma values to be the same. If they are small (< 10),
                # the filter will not have much effect, whereas if they are large (> 150), they will have a very strong effect,
                # making the image look "cartoonish".


            cv2.imwrite("dataset\\%s\\%s.jpg" % (emotion, fileNumber), bilateralFilteredImage)

            fileNumber = fileNumber + 1

    @staticmethod
    def __detectFaceInImage(image):

        # Detect face using 4 different classifiers
        face1 = ImagePreProcessService.faceDet1.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10,
                                                                 minSize=(5, 5),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = ImagePreProcessService.faceDet2.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10,
                                                                 minSize=(5, 5),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = ImagePreProcessService.faceDet3.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10,
                                                                 minSize=(5, 5),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = ImagePreProcessService.faceDet4.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10,
                                                                 minSize=(5, 5),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)

        # Go over detected faces, stop at first detected face, return empty if no face.
        if len(face1) == 1:
            facefeatures = face1
        elif len(face2) == 1:
            facefeatures = face2
        elif len(face3) == 1:
            facefeatures = face3
        elif len(face4) == 1:
            facefeatures = face4
        else:
            "Empty Face"
            facefeatures = ""

        return facefeatures

    @staticmethod
    def __cutFace(facefeatures, grayImage):

        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            grayImage = grayImage[y:y + h, x:x + w]  # Cut the frame to size
            out = cv2.resize(grayImage, (350, 350))

            return out

    @staticmethod
    def __convertImageInGrayScale(image):

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayImage

    @staticmethod
    def __applyImageAveraging(image, kernelSize):
        averagedImage = cv2.blur(image, (kernelSize, kernelSize))
        return averagedImage

    @staticmethod
    def __applyGaussainBlur(image, kernelSize, sigma):
        gaussianBlurredImage = cv2.GaussianBlur(image, (kernelSize, kernelSize), sigma)
        return gaussianBlurredImage

    @staticmethod
    def __applyMedianBlur(image, kernelSize):
        medianBlurredImage = cv2.medianBlur(image, kernelSize)
        return medianBlurredImage

    @staticmethod
    def __applyBilateralFiltering(image, kernelSize, sigma1, sigma2):
        bilateralFilteredImage = cv2.bilateralFilter(image, kernelSize, sigma1, sigma2)
        return bilateralFilteredImage
