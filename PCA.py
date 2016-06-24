import cv2
import numpy as np

class PCA:

    def computeEigenvectors(self, descriptors):
        mean, eigenvectors = cv2.PCACompute(descriptors, mean=np.array([]))
        return eigenvectors