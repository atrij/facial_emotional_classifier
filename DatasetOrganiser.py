import glob
from shutil import copyfile

class DatasetOrganiser:

    @staticmethod
    def organiseDataset(emotions, datasetPathEmotions, datasetPathImages):

        participants = glob.glob(datasetPathEmotions + "/*") #Returns a list of all folders with participant numbers (eg. source_emotions/S005)

        for x in participants:
            part = "%s" % x[-4:]  # store current participant number (eg. S005)

            for sessions in glob.glob("%s/*" % x):  # Store list of sessions for current participant
                # eg. of a session (source_emotions/S005/001)

                for files in glob.glob("%s/*" % sessions):
                    # eg. of a file (source_emotions/S005/001/S005_001_00000011_emotion.txt)

                    current_session = files[20:-30] # (Eg. /001)
                    file = open(files, 'r') # Open a file for read

                    emotion = int(
                        float(file.readline()))  # emotions are encoded as a float, readline as float, then convert to integer.

                    sourcefile_emotion = glob.glob(datasetPathImages + "/%s/%s/*" % (part, current_session))[
                        -1]  # get path for last image in sequence, which contains the emotion

                    sourcefile_neutral = glob.glob(datasetPathImages + "/%s/%s/*" % (part, current_session))[
                        0]  # do same for neutral image

                    dest_neut = "sorted_set\\neutral\\%s" % sourcefile_neutral[25:]  # Generate path to put neutral image
                    dest_emot = "sorted_set\\%s\\%s" % (
                    emotions[emotion], sourcefile_emotion[25:])  # Do same for emotion containing image

                    copyfile(sourcefile_neutral, dest_neut)  # Copy file
                    copyfile(sourcefile_emotion, dest_emot)  # Copy file