import math
from itertools import combinations

import numpy as np

import CentroidUtils as centroidUtils
from LoaderFeatures import FEATURE_NAMES, getFeaturesAsDataFrame

DATA_FRAME = getFeaturesAsDataFrame()
QTY_REQUIRED_FEATURES = 2
selectedFeatures = ()
maxDistance = np.zeros(len(FEATURE_NAMES) - 1)

def getSubset(features):
  lenght = len(features)

  if lenght == QTY_REQUIRED_FEATURES: return ()

  return combinations(features, lenght - 1)

def getCentroidsDistance(features):
  _features = list(features)
  data = DATA_FRAME[_features + ['category']]
  
  haploidDF = data.loc[DATA_FRAME['category'] == 'haploid']
  diploidDF = data.loc[DATA_FRAME['category'] == 'diploid']

  haploidDF = haploidDF[_features]
  diploidDF = diploidDF[_features]

  haploidCentroid = centroidUtils.calculateCentroid(haploidDF[:].values)
  diploidCentroid = centroidUtils.calculateCentroid(diploidDF[:].values)

  return math.dist(haploidCentroid, diploidCentroid)

def isPromisingSolution(features):
  newDistance = getCentroidsDistance(features)

  global maxDistance
  
  if (newDistance > maxDistance[len(features) - 2]):
    maxDistance[len(features) - 2] = newDistance
    return True

  return False

def branchAndBound(features):
  if not isPromisingSolution(features): return
  
  if len(features) == QTY_REQUIRED_FEATURES:
    global selectedFeatures
    selectedFeatures = features
    return

  for subsetFeatures in getSubset(features):
    branchAndBound(subsetFeatures)

def validateBranchAndBound(features):
  maxDistance = 0
  bestCombination = ()

  for combination in list(combinations(features, 2)):
    newDistance = getCentroidsDistance(combination)
    if newDistance > maxDistance:
      maxDistance = newDistance
      bestCombination = combination

  print('validate branch and bound:', bestCombination)

def main():
  branchAndBound(tuple(FEATURE_NAMES))
  print('Características selecionadas:', selectedFeatures)
  print('maxDistance:', maxDistance)

main()