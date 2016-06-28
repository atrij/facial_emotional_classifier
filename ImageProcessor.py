import cv2
from ImageProvider import ImageProvider


class ImageProcessor:

    faceDet1 = cv2.CascadeClassifier("/Users/harsh/opencv/data/haarcascades/haarcascade_frontalface_default.xml")
    faceDet2 = cv2.CascadeClassifier("/Users/harsh/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml")
    faceDet3 = cv2.CascadeClassifier("/Users/harsh/opencv/data/haarcascades/haarcascade_frontalface_alt.xml")
    faceDet4 = cv2.CascadeClassifier("/Users/harsh/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml")

    @staticmethod
    def performPreprocessing(emotion):
        files = ImageProvider.getImages(emotion)

        fileNumber = 0

        for file in files:
            image = cv2.imread(file)

            grayImage = ImageProcessor.convertImageInGrayScale(image)

            facefeatures = ImageProcessor.detectFaceInImage(grayImage)

            outputImage = ImageProcessor.cutFace(facefeatures, grayImage)

            cv2.imwrite("dataset\\%s\\%s.jpg" % (emotion, fileNumber), outputImage)

            fileNumber = fileNumber + 1

    @staticmethod
    def detectFaceInImage(image):

        # Detect face using 4 different classifiers
        face1 = ImageProcessor.faceDet1.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = ImageProcessor.faceDet2.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = ImageProcessor.faceDet3.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = ImageProcessor.faceDet4.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
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
    def cutFace(facefeatures, grayImage):

        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            grayImage = grayImage[y:y + h, x:x + w]  # Cut the frame to size
            out = cv2.resize(grayImage, (350, 350))

            return out

    @staticmethod
    def convertImageInGrayScale(image):

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayImage