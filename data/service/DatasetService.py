import glob
from shutil import copyfile
import random
import cv2

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
            files = glob.glob("dataset\\%s\\*" % emotion)
            random.shuffle(files)

            training = files[:int(len(files) * 0.8)]  # get first 80% of file list
            test = files[-int(len(files) * 0.2):]  # get last 20% of file list

            trainingData[emotion] = training
            testData[emotion] = test

        return trainingData, testData

    @staticmethod
    def getImages(emotion):

        files = glob.glob("sorted_set\\%s\\*" % emotion)
        return files

    @staticmethod
    def getImageDictionaryFromFilePaths(pathList, emotions):

        imageDictionary = {}

        for emotion in emotions:
            paths = pathList[emotion]

            imageList = []
            for path in paths:
                image = cv2.imread(path, 0)

                if (image is not None):
                    imageList.append(image)

            imageDictionary[emotion] = imageList

        return imageDictionary

    @staticmethod
    def __organiseCKDataset(datasetPathEmotions, datasetPathImages, emotions):

        participants = glob.glob(
            datasetPathEmotions + "/*")  # Returns a list of all folders with participant numbers (eg. source_emotions/S005)

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

                    sourcefile_emotion = glob.glob(datasetPathImages + "/%s/%s/*" % (part, current_session))[
                        -1]  # get path for last image in sequence, which contains the emotion

                    sourcefile_neutral = glob.glob(datasetPathImages + "/%s/%s/*" % (part, current_session))[
                        0]  # do same for neutral imageD

                    dest_neut = "sorted_set\\neutral\\%s" % sourcefile_neutral[
                                                            25:]  # Generate path to put neutral image
                    dest_emot = "sorted_set\\%s\\%s" % (
                        emotions[emotion], sourcefile_emotion[25:])  # Do same for emotion containing image

                    copyfile(sourcefile_neutral, dest_neut)  # Copy file
                    copyfile(sourcefile_emotion, dest_emot)  # Copy file