import glob
import random
import cv2

class ImageProvider:

    @staticmethod
    def getImages(emotion):

        files = glob.glob("sorted_set\\%s\\*" % emotion)
        return files

    @staticmethod
    def splitDataset(emotions):

        trainingData = {}
        testData = {}

        for emotion in emotions:
            files = glob.glob("dataset\\%s\\*" % emotion)
            random.shuffle(files)

            training = files[:int(len(files) * 0.8)]  # get first 80% of file list
            test = files[-int(len(files) * 0.2):]  # get last 20% of file list

            trainingData[emotion] = training
            testData[emotion] = test

        return trainingData, testData

    @staticmethod
    def getImageDictionaryFromFilePaths(pathList, emotions):

        imageDictionary = {}

        for emotion in emotions:
            paths = pathList[emotion]

            imageList = []
            for path in paths:
                image = cv2.imread(path, 0)

                if(image is not None):
                    imageList.append(image)

            imageDictionary[emotion] = imageList

        return imageDictionary
