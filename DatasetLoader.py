import itertools
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize

import CentroidUtils as centroidUtils
from NormalizationUtils import logNormalization, minMaxNormalization


def getAngle(a, b, c):
  # print("angle:")
  # print(a, b, c)
  _a = np.array(a)
  _b = np.array(b)
  _c = np.array(c)

  ba = _a - _b
  bc = _c - _b

  cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
  angle = np.arccos(cosine_angle)

  # print(np.degrees(angle))
  return np.degrees(angle)


def createInstance(points, target):
  combinations = list(itertools.combinations(points, 2))
  instance = {}

  for pointA, pointB in combinations:
    key = "d{}{}".format(points.index(pointA) + 1, points.index(pointB) + 1)
    distance = math.dist(pointA, pointB)
    instance[key] = distance
  
  centroid = centroidUtils.calculateCentroid(points)
  centroidSize = centroidUtils.calculateCentroidSize(points, centroid)
  
  instance["CS"] = centroidSize
  instance["a213"] = getAngle(points[1], points[0], points[2])
  instance["a134"] = getAngle(points[0], points[2], points[3])
  instance["a436"] = getAngle(points[3], points[2], points[5])
  instance["a367"] = getAngle(points[2], points[5], points[6])
  instance["a768"] = getAngle(points[6], points[5], points[7])
  instance["a679"] = getAngle(points[5], points[6], points[8])

  instance["target"] = target
  return instance

def getInstances(annotationFile, target):
  annotations = json.load(open(os.path.join('./', annotationFile)))
  image_metadata = annotations['_via_img_metadata']
  image_id_list =  annotations['_via_image_id_list']

  instances = []

  for keyFilename in image_id_list:
    # print(keyFilename)
    regions = image_metadata[keyFilename]['regions']

    points = []
    for i in range(len(regions)):
      region = regions[i]['shape_attributes']
      point = (region['cx'], region['cy'])
      points.append(point)

    instance = createInstance(points, target)
    instances.append(instance)

  return instances

def getInstancesAsDataFrame(annotationFile, target):
  instances = getInstances(annotationFile, target)
  dataFrame = pd.DataFrame(instances)
  return dataFrame

def normalizeDataset(dataset, featureNames):
  for feature in featureNames:
    dataset[feature] = normalize([dataset[feature]], norm='max')[0]
    # dataset[feature] = MinMaxScaler().fit_transform(np.array(dataset[feature]).reshape(-1, 1)) 
    # dataset[feature] = logNormalization(dataset[feature])
  
  return dataset

def loadDataset(normalized = True):
  # pd.set_option('display.max_rows', None)
  diploidDf = getInstancesAsDataFrame('diploid.json', 'diploid')
  diploidTestDf = getInstancesAsDataFrame('diploid_test.json', 'diploid')
  
  haploidDf = getInstancesAsDataFrame('haploid.json', 'haploid')
  haploidTestDf = getInstancesAsDataFrame('haploid_test.json', 'haploid')

  dataset = pd.concat([diploidDf, diploidTestDf, haploidDf, haploidTestDf], axis = 0, ignore_index = True)
  featureNames = dataset.columns.difference(['target'])

  if normalized:
    dataset = normalizeDataset(dataset, featureNames)

  return dataset, featureNames

# DATA_SET, FEATURE_NAMES = loadDataset()
# print(DATA_SET)
# print(len(FEATURE_NAMES))
# print(DATA_SET[['a134', 'a213', 'a367', 'a436', 'a679', 'a768']])
# print(DATA_SET["a213","a134","a436","a367","a768","a679"])

