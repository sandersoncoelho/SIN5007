import math
from itertools import combinations

import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

import CentroidUtils as centroidUtils
from DatasetLoader import loadDataset

DATA_SET, FEATURE_NAMES = loadDataset()
QTY_REQUIRED_FEATURES = 2
FIRST_CLASS = 'diploid'
SECOND_CLASS = 'haploid'
selectedFeatures = ()
maxDistance = np.zeros(len(FEATURE_NAMES) - 1)

def getSubset(features):
  lenght = len(features)

  if lenght == QTY_REQUIRED_FEATURES: return ()

  return combinations(features, lenght - 1)

def getCentroidsDistance(features):
  _features = list(features)
  data = DATA_SET[_features + ['target']]
  
  firstClassDF = data.loc[DATA_SET['target'] == FIRST_CLASS]
  secondClassDF = data.loc[DATA_SET['target'] == SECOND_CLASS]

  firstClassDF = firstClassDF[_features]
  secondClassDF = secondClassDF[_features]

  firstClassCentroid = centroidUtils.calculateCentroid(firstClassDF[:].values)
  secondClassCentroid = centroidUtils.calculateCentroid(secondClassDF[:].values)

  return math.dist(firstClassCentroid, secondClassCentroid)

def isPromisingSolution(features):
  newDistance = getCentroidsDistance(features)

  global maxDistance
  
  if newDistance > maxDistance[len(features) - 2]:
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

def sequentialBackwardSelection():
  X = DATA_SET[FEATURE_NAMES]
  y = DATA_SET["target"]

  knn = KNeighborsClassifier(n_neighbors = QTY_REQUIRED_FEATURES)
  sfs = SequentialFeatureSelector(knn, direction='backward', n_features_to_select = QTY_REQUIRED_FEATURES)
  sfs.fit(X, y)
  print(sfs.get_support())
  print(sfs.get_feature_names_out())

def main():
  branchAndBound(tuple(FEATURE_NAMES))
  # sequentialBackwardSelection()
  # validateBranchAndBound(FEATURE_NAMES)
  print('Características selecionadas:', selectedFeatures)
  print('maxDistance:', maxDistance)

main()