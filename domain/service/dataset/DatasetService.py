import glob
import random
import cv2

from data.service.Database import Database
from domain.service.config.Constants import Constants


class DatasetService:

    @staticmethod
    def organiseDataset(datasetName, emotions, datasetPathEmotions, datasetPathImages):

        if(datasetName == Constants.cohn_Kanade_extended):
            print "Cohn Kanade Extended Dataset Detected"
            DatasetService.__organiseCKDataset(datasetPathEmotions, datasetPathImages, emotions)

    @staticmethod
    def splitDataset(emotions):

        trainingData = {}
        testData = {}

        for emotion in emotions:
            files = Database.getProcessedFiles(emotion)
            print ("Number of images at time of splitting dataset for %s is %d", (emotion, len(files)))
            random.shuffle(files)

            training = files[:int(len(files) * 0.8)]  # get first 80% of file list
            test = files[-int(len(files) * 0.2):]  # get last 20% of file list

            trainingData[emotion] = training
            testData[emotion] = test

            print ("Number of training images for %s is %d ", (emotion, len(training)))

        return trainingData, testData

    @staticmethod
    def getImages(emotion):

        files = Database.getImages(emotion)
        return files

    @staticmethod
    def getImageDictionaryFromFilePaths(pathDictionary, emotions):

        imageDictionary = {}

        for emotion in emotions:
            pathList = pathDictionary[emotion]

            print ("Number of images before making imageDictionary for %s is %d", (emotion, len(pathList)))

            imageList = []
            for path in pathList:
                image = cv2.imread(path, 0)

                if (image is not None):
                    imageList.append(image)

            imageDictionary[emotion] = imageList
            print ("Number of images after making imageDictionary for %s is %d", (emotion, len(imageDictionary[emotion])))

        return imageDictionary

    @staticmethod
    def __organiseCKDataset(datasetPathEmotions, datasetPathImages, emotions):

        participants = Database.getAllParticipants(datasetPathEmotions)

        for x in participants:
            part = "%s" % x[-4:]  # store current participant number (eg. S005)

            for sessions in glob.glob("%s/*" % x):  # Store list of sessions for current participant
                # eg. of a session (source_emotions/S005/001)

                for files in glob.glob("%s/*" % sessions):
                    # eg. of a file (source_emotions/S005/001/S005_001_00000011_emotion.txt)

                    current_session = files[20:-30]  # (Eg. /001)
                    file = open(files, 'r')  # Open a file for read

                    emotion = int(
                        float(
                            file.readline()))  # emotions are encoded as a float, readline as float, then convert to integer.

                    sourcefile_emotion = Database.getPathForImageContainingFullEmotion(datasetPathImages, part, current_session)

                    sourcefile_neutral = Database.getPathForImageContainingNoEmotion(datasetPathImages, part, current_session)

                    dest_neut = Database.generatePathForNeutralImage(sourcefile_neutral)
                    dest_emot = Database.generatePathForFullEmotionImage(emotions, emotion, sourcefile_emotion)

                    Database.copyFiles(sourcefile_neutral, sourcefile_emotion, dest_neut, dest_emot)