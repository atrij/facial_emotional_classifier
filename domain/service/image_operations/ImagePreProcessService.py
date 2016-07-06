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

        # Check if grayScale conversion is required
        if Constants.grayScaleConversion in methodList:
            grayScaleConversion = True

        # Check if face detection is required
        if Constants.faceDetectionHAAR in methodList:
            faceDetectionHAAR = True

        if Constants.averaging in methodList:
            averaging = True

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
                averagedImage = cv2.blur(outputImage, (5,5))

            cv2.imwrite("dataset\\%s\\%s.jpg" % (emotion, fileNumber), averagedImage)

            fileNumber = fileNumber + 1

    @staticmethod
    def __detectFaceInImage(image):

        # Detect face using 4 different classifiers
        face1 = ImagePreProcessService.faceDet1.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = ImagePreProcessService.faceDet2.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = ImagePreProcessService.faceDet3.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = ImagePreProcessService.faceDet4.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
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