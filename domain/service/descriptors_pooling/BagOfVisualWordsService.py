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

            eigenvectorListList = eigenVectorsDictionary[emotion]
            histogramArrayList = []

            for eigenvectorList in eigenvectorListList:

                img_clustered_words = kMeans.predict(eigenvectorList) # Each descriptor is mapped to a word

                histogram = np.array(
                    [np.bincount(img_clustered_words, minlength=self.clusterCount)]) # Create a histogram ; how many times does a word (cluster) come in an image?

                histogramArray = np.array(histogram)

                histogramArrayList.append(histogramArray)

            histogramDictionary[emotion] = histogramArrayList

        print ("Length of HistogramArrayList for one emotion - ", len(histogramDictionary["contempt"]))
        print ("Length of HistogramArray for a random image in one emotion - ", len(histogramDictionary["contempt"][0]))
        return histogramDictionary
