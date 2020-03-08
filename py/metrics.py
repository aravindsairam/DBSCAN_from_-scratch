import pandas as pd
import numpy as np
import math
import time

def EuclideanDistance(p1, p2):
    return  np.sqrt(np.sum(np.square(p1 - p2)))

def adjustRandIndex(trueLabel, predictLabel, size):
    """
    @param trueLabel : true label from the dataset
    @param predictLabel : predicted label from DBSCAN
    @param size : length of the dataset
    @return adjusted Rand Index with function runtime
    """
    sT = time.time()
    def combination(a):
        if a < 2:
            return 0
        fact = math.factorial(a)/(2*math.factorial(a-2))
        return fact

    # Contingency table
    contTable = pd.crosstab(trueLabel, predictLabel)
    nij = 0
    for rows in range(len(contTable)):
        nij+= np.sum(list(map(combination, contTable.iloc[rows])))

    contTable['sumCols'] = contTable.sum(axis = 1)
    contTable['sumrows'] = contTable.sum(axis = 0)
    contTable.fillna(0, inplace = True)

    ai = np.sum(list(map(combination, contTable['sumrows'])))
    bi = np.sum(list(map(combination, contTable['sumCols'])))

    abi = ai * bi
    num = nij - (abi/combination(size))
    denom = (1/2*(ai+bi)) - (abi/combination(size))
    ARI = num/denom
    eT = time.time()
    return ARI, eT-sT


def silhouetteScore(X, y, numClusters):
    """
    @param X : Input features
    @param y : predicted label from DBSCAN
    @param numClusters : number of predicted clusters
    @return silhouette coefficient with function runtime
    """
    sT = time.time()
    if numClusters <= 1:
        meanScore = 0
    else:
        # mean distance between i and all other data points in the same cluster
        meanSimilarity = []
        for nC in range(1, numClusters+1):
            tempData = X[y == nC]
            for l1 in range(len(tempData)):
                tempSum = 0.0
                tempRow = tempData.iloc[l1].values
                for l2 in range(len(tempData)):
                    tempSum += EuclideanDistance(tempRow, tempData.iloc[l2].values)
                tempSum = tempSum/(len(tempData)-1)
                meanSimilarity.append(tempSum)
        # mean dissimilarity of point i to some cluster C
        # as the mean of the distance from ito all points in C
        meanDissimilarity = []
        for nC1 in range(1, numClusters+1):
            tempData1 = X[y == nC1]
            for l1 in range(len(tempData1)):
                tempRow = tempData1.iloc[l1].values
                minDist = []
                for nC2 in range(1, numClusters+1):
                    if nC1 != nC2:
                        tempData2 = X[y == nC2]
                        tempSum = 0.0
                        for l2 in range(len(tempData2)):
                            tempSum += EuclideanDistance(tempRow, tempData2.iloc[l2].values)
                        tempSum = tempSum/(len(tempData2))
                        minDist.append(tempSum)
                meanDissimilarity.append(min(minDist))
        score = pd.DataFrame({'mean_sim':meanSimilarity,
                         'mean_disim':meanDissimilarity})
        score['silhouette_score'] = (score['mean_disim'] - score['mean_sim'])/ score.max(axis = 1)
        meanScore = score['silhouette_score'].mean()
    eT = time.time()
    return meanScore, eT-sT
