from domain.service.config.Constants import Constants
from domain.service.dimensionality_reduction.PrincipalComponentAnalysisService import PrincipalComponentAnalysis


class DimensionalityReducer:

    @staticmethod
    def reduceDimensionality(modelName, argumentList):

        if(modelName == Constants.pca):
            eigenvectorsDictionary = PrincipalComponentAnalysis.computeEigenvectors(argumentList[0], argumentList[1])
            print ("Length of Eigenvector List for one emotion - ", len(eigenvectorsDictionary["contempt"]))
            print ("One eigenvector for contempt - ", eigenvectorsDictionary["contempt"][0])
            return eigenvectorsDictionary