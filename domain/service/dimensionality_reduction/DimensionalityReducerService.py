from domain.service.config.Constants import Constants
from domain.service.dimensionality_reduction.PrincipalComponentAnalysisService import PrincipalComponentAnalysisService


class DimensionalityReducerService:

    @staticmethod
    def reduceDimensionality(modelName, argumentList):

        if(modelName == Constants.pca):
            eigenvectorsDictionary = PrincipalComponentAnalysisService.computeEigenvectors(argumentList[0], argumentList[1])
            print ("Length of EigenvectorListList for one emotion - ", len(eigenvectorsDictionary["contempt"]))
            print ("Length of EigenvectorList for a random image in one emotion - ", len(eigenvectorsDictionary["contempt"][0]))
            return eigenvectorsDictionary