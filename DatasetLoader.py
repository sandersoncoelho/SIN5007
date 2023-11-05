import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import CentroidUtils as centroidUtils
from NormalizationUtils import minMaxNormalization

FEATURE_NAMES = ['d12', 'd13', 'd34', 'd36', 'd45', 'd67', 'd68', 'd79', 'd810', 'd910', 'CS']

class Instance:
  def __init__(self, d12 = None, d13 = None, d34 = None, d36 = None, d45 = None, d67 = None, d68 = None, d79 = None, d810 = None, d910 = None, centroidSize = None):
    self.d12 = d12
    self.d13 = d13
    self.d34 = d34
    self.d36 = d36
    self.d45 = d45
    self.d67 = d67
    self.d68 = d68
    self.d79 = d79
    self.d810 = d810
    self.d910 = d910
    self.centroidSize = centroidSize


def getInstances(annotationFile):
  annotations = json.load(open(os.path.join('./', annotationFile)))
  image_metadata = annotations['_via_img_metadata']
  image_id_list =  annotations['_via_image_id_list']

  instances = []

  for keyFilename in image_id_list:
    regions = image_metadata[keyFilename]['regions']
    
    point1 = regions[0]['shape_attributes']
    point2 = regions[1]['shape_attributes']
    point3 = regions[2]['shape_attributes']
    point4 = regions[3]['shape_attributes']
    point5 = regions[4]['shape_attributes']
    point6 = regions[5]['shape_attributes']
    point7 = regions[6]['shape_attributes']
    point8 = regions[7]['shape_attributes']
    point9 = regions[8]['shape_attributes']
    point10 = regions[9]['shape_attributes']

    p1 = [point1['cx'], point1['cy']]
    p2 = [point2['cx'], point2['cy']]
    p3 = [point3['cx'], point3['cy']]
    p4 = [point4['cx'], point4['cy']]
    p5 = [point5['cx'], point5['cy']]
    p6 = [point6['cx'], point6['cy']]
    p7 = [point7['cx'], point7['cy']]
    p8 = [point8['cx'], point8['cy']]
    p9 = [point9['cx'], point9['cy']]
    p10 = [point10['cx'], point10['cy']]
    centroid = centroidUtils.calculateCentroid([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
    centroidSize = centroidUtils.calculateCentroidSize([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10], centroid)

    instance = Instance()
    instance.d12 = math.dist(p1, p2)
    instance.d13 = math.dist(p1, p3)
    instance.d34 = math.dist(p3, p4)
    instance.d36 = math.dist(p3, p6)
    instance.d45 = math.dist(p4, p5)
    instance.d67 = math.dist(p6, p7)
    instance.d68 = math.dist(p6, p8)
    instance.d79 = math.dist(p7, p9)
    instance.d810 = math.dist(p8, p10)
    instance.d910 = math.dist(p9, p10)
    instance.centroidSize = centroidSize

    instances.append(instance)

  return instances

def getInstancesAsDataFrame(annotationFile, target):
  instances = getInstances(annotationFile)

  featureD12 = []
  featureD13 = []
  featureD34 = []
  featureD36 = []
  featureD45 = []
  featureD67 = []
  featureD68 = []
  featureD79 = []
  featureD810 = []
  featureD910 = []
  featureCentroidSize = []
  targetList = []

  for instance in instances:
    featureD12.append(instance.d12)
    featureD13.append(instance.d13)
    featureD34.append(instance.d34)
    featureD36.append(instance.d36)
    featureD45.append(instance.d45)
    featureD67.append(instance.d67)
    featureD68.append(instance.d68)
    featureD79.append(instance.d79)
    featureD810.append(instance.d810)
    featureD910.append(instance.d910)
    featureCentroidSize.append(instance.centroidSize)
    targetList.append(target)

  data = {
    'd12': featureD12,
    'd13': featureD13,
    'd34': featureD34,
    'd36': featureD36,
    'd45': featureD45,
    'd67': featureD67,
    'd68': featureD68,
    'd79': featureD79,
    'd810': featureD810,
    'd910': featureD910,
    'CS': featureCentroidSize,
    'target': targetList
  }

  dataFrame = pd.DataFrame(data)
  return dataFrame

def normalizeDataset(dataset):
  for feature in FEATURE_NAMES:
    dataset[feature] = MinMaxScaler().fit_transform(np.array(dataset[feature]).reshape(-1,1)) 
  
  return dataset

def loadDataset(normalized = True):
  pd.set_option('display.max_rows', None)
  diploidDf = getInstancesAsDataFrame('diploid.json', 'diploid')
  diploidTestDf = getInstancesAsDataFrame('diploid_test.json', 'diploid')
  
  haploidDf = getInstancesAsDataFrame('haploid.json', 'haploid')
  haploidTestDf = getInstancesAsDataFrame('haploid_test.json', 'haploid')

  dataset = pd.concat([diploidDf, diploidTestDf, haploidDf, haploidTestDf], axis=0, ignore_index=True)

  if normalized:
    dataset = normalizeDataset(dataset)

  return dataset
