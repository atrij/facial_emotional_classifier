from domain.service.config.Constants import Constants
from domain.service.dimensionality_reduction.PrincipalComponentAnalysisService import PrincipalComponentAnalysis


class DimensionalityReducer:

    @staticmethod
    def reduceDimensionality(modelName, argumentList):

        if(modelName == Constants.pca):
            eigenvectorsDictionary = PrincipalComponentAnalysis.computeEigenvectors(argumentList[0], argumentList[1])
            return eigenvectorsDictionary