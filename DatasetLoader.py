import itertools
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import CentroidUtils as centroidUtils


def createInstance(points, centroidSize, target):
  combinations = list(itertools.combinations(points, 2))
  instance = {}

  for pointA, pointB in combinations:
    key = "d{}{}".format(points.index(pointA) + 1, points.index(pointB) + 1)
    distance = math.dist(pointA, pointB)
    instance[key] = distance
  
  instance["CS"] = centroidSize
  instance["target"] = target
  return instance

def getInstances(annotationFile, target):
  annotations = json.load(open(os.path.join('./', annotationFile)))
  image_metadata = annotations['_via_img_metadata']
  image_id_list =  annotations['_via_image_id_list']

  instances = []

  for keyFilename in image_id_list:
    regions = image_metadata[keyFilename]['regions']

    points = []
    for i in range(len(regions)):
      region = regions[i]['shape_attributes']
      point = (region['cx'], region['cy'])
      points.append(point)
    
    centroid = centroidUtils.calculateCentroid(points)
    centroidSize = centroidUtils.calculateCentroidSize(points, centroid)

    instance = createInstance(points, centroidSize, target)
    instances.append(instance)

  return instances

def getInstancesAsDataFrame(annotationFile, target):
  instances = getInstances(annotationFile, target)
  dataFrame = pd.DataFrame(instances)
  return dataFrame

def normalizeDataset(dataset, featureNames):
  for feature in featureNames:
    dataset[feature] = MinMaxScaler().fit_transform(np.array(dataset[feature]).reshape(-1, 1)) 
  
  return dataset

def loadDataset(normalized = True):
  pd.set_option('display.max_rows', None)
  diploidDf = getInstancesAsDataFrame('diploid.json', 'diploid')
  diploidTestDf = getInstancesAsDataFrame('diploid_test.json', 'diploid')
  
  haploidDf = getInstancesAsDataFrame('haploid.json', 'haploid')
  haploidTestDf = getInstancesAsDataFrame('haploid_test.json', 'haploid')

  dataset = pd.concat([diploidDf, diploidTestDf, haploidDf, haploidTestDf], axis = 0, ignore_index = True)
  featureNames = dataset.columns.difference(['target'])

  if normalized:
    dataset = normalizeDataset(dataset, featureNames)

  return dataset, featureNames
