import cv2
import numpy as np

class PrincipalComponentAnalysis:

    @staticmethod
    def computeEigenvectors(descriptorsDictionary, emotions):

        eigenvectorDictionary = {}

        for emotion in emotions:

            eigenvectorsList = []
            descriptorsList = descriptorsDictionary[emotion]

            for descriptors in descriptorsList:
                mean, eigenvectors = cv2.PCACompute(descriptors, mean=np.array([]))
                eigenvectorsList.append(eigenvectors)

            eigenvectorDictionary[emotion] = eigenvectorsList

        return eigenvectorDictionary