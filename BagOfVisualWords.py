import numpy as np
from sklearn.cluster import KMeans

class BagOfVisualWords:

    def __init__(self, clusterCount, trainingDescriptorsList):
        self.clusterCount = clusterCount
        self.trainingDescriptorsList = trainingDescriptorsList

    def getHistogramForImages(self, eigenVectorsList):
        kMeans = KMeans(100)
        kMeans.fit(self.trainingDescriptorsList) # Cluster the descriptors to form words

        for eigenvector in eigenVectorsList:
            img_clustered_words = kMeans.predict(eigenvector) # Each descriptor is mapped to a word
            print len(img_clustered_words)

            histogram = np.array(
                [np.bincount(img_clustered_words, minlength=self.clusterCount)]) # Create a histogram ; how many times does a word (cluster) come in an image?

            arr = np.array(histogram)
            print arr.shape