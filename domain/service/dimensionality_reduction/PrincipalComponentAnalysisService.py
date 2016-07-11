import cv2
import numpy as np

class PrincipalComponentAnalysisService:

    @staticmethod
    def computeEigenvectors(descriptorsDictionary, emotions):

        eigenvectorDictionary = {}

        for emotion in emotions:

            eigenvectorListList = []
            descriptorListList = descriptorsDictionary[emotion]

            for descriptorList in descriptorListList:
                mean, eigenvectorList = cv2.PCACompute(descriptorList, mean=np.array([]))
                eigenvectorListList.append(eigenvectorList)

            eigenvectorDictionary[emotion] = eigenvectorListList

        return eigenvectorDictionary