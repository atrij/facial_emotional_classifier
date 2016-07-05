import glob
from shutil import copyfile


class Database:

    @staticmethod
    def getAllParticipants(datasetPathEmotions):

        participants = glob.glob(
            datasetPathEmotions + "/*")  # Returns a list of all folders with participant numbers (eg. source_emotions/S005)

        return participants

    @staticmethod
    def getPathForImageContainingFullEmotion(datasetPathImages, part, current_session):
       path =  glob.glob(datasetPathImages + "/%s/%s/*" % (part, current_session))[
            -1]  # get path for last image in sequence, which contains the emotion

       return path

    @staticmethod
    def getPathForImageContainingNoEmotion(datasetPathImages, part, current_session):
        path = glob.glob(datasetPathImages + "/%s/%s/*" % (part, current_session))[
            0]  # get path for last image in sequence, which contains the emotion

        return path

    @staticmethod
    def generatePathForNeutralImage(sourcefile_neutral):
        dest_neut = "sorted_set\\neutral\\%s" % sourcefile_neutral[
                                                25:]  # Generate path to put neutral image

        return dest_neut

    @staticmethod
    def generatePathForFullEmotionImage(emotions, emotion, sourcefile_emotion):
        dest_emot = "sorted_set\\%s\\%s" % (
            emotions[emotion], sourcefile_emotion[25:])  # Do same for emotion containing image

        return dest_emot

    @staticmethod
    def copyFiles(source_neutral, source_emotional, dest_neutral, dest_emotional):
        copyfile(source_neutral, dest_neutral)  # Copy file
        copyfile(source_emotional, dest_emotional)  # Copy file

    @staticmethod
    def getImages(emotion):
        files = glob.glob("sorted_set\\%s\\*" % emotion)
        return files

    @staticmethod
    def getProcessedFiles(emotion):
        files = glob.glob("dataset\\%s\\*" % emotion)
        return files