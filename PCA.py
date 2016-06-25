import cv2
import numpy as np

class PCA:

    @staticmethod
    def computeEigenvectors(descriptorsList):

        eigenvectorsList = []

        for descriptors in descriptorsList:
            mean, eigenvectors = cv2.PCACompute(descriptors, mean=np.array([]))
            eigenvectorsList.append(eigenvectors)

        return  eigenvectorsList