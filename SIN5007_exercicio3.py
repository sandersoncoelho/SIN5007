import math

import matplotlib.pyplot as plt
import numpy as np

import CentroidUtils as centroidUtils
import LoaderFeatures as loader

haploidFeatureD12, haploidFeatureD13, haploidFeatureD34, haploidFeatureD36, haploidFeatureD45, \
  haploidFeatureD67, haploidFeatureD68, haploidFeatureD79, haploidFeatureD810, haploidFeatureD910, \
    haploidCentroidSize = loader.getFeatures('haploid.json')

amountOfHaploides = len(haploidFeatureD12)

diploidFeatureD12, diploidFeatureD13, diploidFeatureD34, diploidFeatureD36, diploidFeatureD45, \
  diploidFeatureD67, diploidFeatureD68, diploidFeatureD79, diploidFeatureD810, diploidFeatureD910, \
    diploidCentroidSize = loader.getFeatures('diploid.json')

amountOfDiploides = len(diploidFeatureD12)

FEATURE_MAPPER = {
  'd11': (haploidFeatureD12, diploidFeatureD12),
  'd13': (haploidFeatureD13, diploidFeatureD13),
  'd34': (haploidFeatureD34, diploidFeatureD34),
  'd36': (haploidFeatureD36, diploidFeatureD36),
  'd45': (haploidFeatureD45, diploidFeatureD45),
  'd67': (haploidFeatureD67, diploidFeatureD67),
  'd68': (haploidFeatureD68, diploidFeatureD68),
  'd79': (haploidFeatureD79, diploidFeatureD79),
  'd810': (haploidFeatureD810, diploidFeatureD810),
  'd910': (haploidFeatureD910, diploidFeatureD910),
  'CS': (haploidCentroidSize, diploidCentroidSize)
}

selectedFeatures = []
requiredAmountFeatures = 2
maxDistance = 0

def getSubset(features):
  if len(features) == requiredAmountFeatures: return []

  subsets = []

  for i in range(len(features)):
    subset = features.copy()
    subset.pop(i)
    subsets.append(subset)
  return subsets

def isPromisingSolution(features):
  print('Características: ', features)
  haploidInstance = []

  for i in range(amountOfHaploides):
    haploidInstance.append([])
  
  for feature in features:
    haploidFeature = FEATURE_MAPPER[feature][0]
    for i in range(amountOfHaploides):
      haploidInstance[i].append(haploidFeature[i])

  haploidCentroid = centroidUtils.calculateCentroid(haploidInstance)


  diploidInstance = []

  for i in range(amountOfDiploides):
    diploidInstance.append([])
  
  for feature in features:
    diploidFeature = FEATURE_MAPPER[feature][1]
    for i in range(amountOfDiploides):
      diploidInstance[i].append(diploidFeature[i])

  diploidCentroid = centroidUtils.calculateCentroid(diploidInstance)

  
  newDistance = math.dist(haploidCentroid, diploidCentroid)
  global maxDistance
  
  if (newDistance > maxDistance):
    maxDistance = newDistance
    print('Distancia entre os centroides: ', maxDistance)
    return True

  return False

def branchAndBound(features):
  if not isPromisingSolution(features): return

  global selectedFeatures
  selectedFeatures = features
  
  if len(features) == requiredAmountFeatures:
    return

  for subsetFeatures in getSubset(features):
    branchAndBound(subsetFeatures)


branchAndBound(list(FEATURE_MAPPER.keys()))
print('Características selecionadas:', selectedFeatures)
