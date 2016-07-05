import numpy as np
from sklearn.cluster import KMeans

class BagOfVisualWords:

    def __init__(self, clusterCount, trainingDescriptorsList):
        self.clusterCount = clusterCount
        self.trainingDescriptorsList = trainingDescriptorsList

    def getHistogramForImages(self, eigenVectorsDictionary, emotions):
        kMeans = KMeans(self.clusterCount)
        kMeans.fit(self.trainingDescriptorsList) # Cluster the descriptors to form words

        histogramDictionary = {}

        for emotion in emotions:

            eigenvectorsList = eigenVectorsDictionary[emotion]
            histogramList = []

            for eigenvectors in eigenvectorsList:

                img_clustered_words = kMeans.predict(eigenvectors) # Each descriptor is mapped to a word

                histogram = np.array(
                    [np.bincount(img_clustered_words, minlength=self.clusterCount)]) # Create a histogram ; how many times does a word (cluster) come in an image?

                histogramArray = np.array(histogram)

                histogramList.append(histogramArray)

            histogramDictionary[emotion] = histogramList

        print ("Length of Histogram List for one emotion - ", len(histogramDictionary["contempt"]))
        print ("One histogram for contempt - ", histogramDictionary["contempt"][0])
        return histogramDictionary
