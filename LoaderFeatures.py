import json
import math
import os

import numpy as np
import pandas as pd

import CentroidUtils as centroidUtils
import LoaderFeatures as loader
from NormalizationUtils import minMaxNormalization

FEATURE_NAMES = ['d12', 'd13', 'd34', 'd36', 'd45', 'd67', 'd68', 'd79', 'd810', 'd910', 'CS']

class Feature:
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


def getFeatures(annotationFile):
  annotations = json.load(open(os.path.join('./', annotationFile)))
  image_metadata = annotations['_via_img_metadata']
  image_id_list =  annotations['_via_image_id_list']

  d12Feature = np.array([])
  d13Feature = np.array([])
  d34Feature = np.array([])
  d36Feature = np.array([])
  d45Feature = np.array([])
  d67Feature = np.array([])
  d68Feature = np.array([])
  d79Feature = np.array([])
  d810Feature = np.array([])
  d910Feature = np.array([])
  centroidSizeFeature = np.array([])

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

    d12Feature = np.append(d12Feature, math.dist(p1, p2))
    d13Feature = np.append(d13Feature, math.dist(p1, p3))
    d34Feature = np.append(d34Feature, math.dist(p3, p4))
    d36Feature = np.append(d36Feature, math.dist(p3, p6))
    d45Feature = np.append(d45Feature, math.dist(p4, p5))
    d67Feature = np.append(d67Feature, math.dist(p6, p7))
    d68Feature = np.append(d68Feature, math.dist(p6, p8))
    d79Feature = np.append(d79Feature, math.dist(p7, p9))
    d810Feature = np.append(d810Feature, math.dist(p8, p10))
    d910Feature = np.append(d910Feature, math.dist(p9, p10))
    centroidSizeFeature = np.append(centroidSizeFeature, centroidSize)

  return d12Feature, d13Feature, d34Feature, d36Feature, d45Feature, \
    d67Feature, d68Feature, d79Feature, d810Feature, d910Feature, \
    centroidSizeFeature

def getFeaturesAsDataFrame():
  haploidFeatureD12, haploidFeatureD13, haploidFeatureD34, haploidFeatureD36, haploidFeatureD45, \
    haploidFeatureD67, haploidFeatureD68, haploidFeatureD79, haploidFeatureD810, haploidFeatureD910, \
      haploidCentroidSize = loader.getFeatures('haploid.json')

  diploidFeatureD12, diploidFeatureD13, diploidFeatureD34, diploidFeatureD36, diploidFeatureD45, \
    diploidFeatureD67, diploidFeatureD68, diploidFeatureD79, diploidFeatureD810, diploidFeatureD910, \
      diploidCentroidSize = loader.getFeatures('diploid.json')

  normHaploidFeatureD12 = minMaxNormalization(haploidFeatureD12)
  normHaploidFeatureD13 = minMaxNormalization(haploidFeatureD13)
  normHaploidFeatureD34 = minMaxNormalization(haploidFeatureD34)
  normHaploidFeatureD45 = minMaxNormalization(haploidFeatureD45)
  normHaploidFeatureD36 = minMaxNormalization(haploidFeatureD36)
  normHaploidFeatureD67 = minMaxNormalization(haploidFeatureD67)
  normHaploidFeatureD68 = minMaxNormalization(haploidFeatureD68)
  normHaploidFeatureD79 = minMaxNormalization(haploidFeatureD79)
  normHaploidFeatureD810 = minMaxNormalization(haploidFeatureD810)
  normHaploidFeatureD910 = minMaxNormalization(haploidFeatureD910)
  normHaploidCentroidSize = minMaxNormalization(haploidCentroidSize)

  normDiploidFeatureD12 = minMaxNormalization(diploidFeatureD12)
  normDiploidFeatureD13 = minMaxNormalization(diploidFeatureD13)
  normDiploidFeatureD34 = minMaxNormalization(diploidFeatureD34)
  normDiploidFeatureD36 = minMaxNormalization(diploidFeatureD36)
  normDiploidFeatureD45 = minMaxNormalization(diploidFeatureD45)
  normDiploidFeatureD67 = minMaxNormalization(diploidFeatureD67)
  normDiploidFeatureD68 = minMaxNormalization(diploidFeatureD68)
  normDiploidFeatureD79 = minMaxNormalization(diploidFeatureD79)
  normDiploidFeatureD810 = minMaxNormalization(diploidFeatureD810)
  normDiploidFeatureD910 = minMaxNormalization(diploidFeatureD910)
  normDiploidCentroidSize = minMaxNormalization(diploidCentroidSize)

  featureD12 = np.concatenate((normHaploidFeatureD12, normDiploidFeatureD12))
  featureD13 = np.concatenate((normHaploidFeatureD13, normDiploidFeatureD13))
  featureD34 = np.concatenate((normHaploidFeatureD34, normDiploidFeatureD34))
  featureD36 = np.concatenate((normHaploidFeatureD36, normDiploidFeatureD36))
  featureD45 = np.concatenate((normHaploidFeatureD45, normDiploidFeatureD45))
  featureD67 = np.concatenate((normHaploidFeatureD67, normDiploidFeatureD67))
  featureD68 = np.concatenate((normHaploidFeatureD68, normDiploidFeatureD68))
  featureD79 = np.concatenate((normHaploidFeatureD79, normDiploidFeatureD79))
  featureD810 = np.concatenate((normHaploidFeatureD810, normDiploidFeatureD810))
  featureD910 = np.concatenate((normHaploidFeatureD910, normDiploidFeatureD910))
  centroidSizeFeature = np.concatenate((normHaploidCentroidSize, normDiploidCentroidSize))

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
    'CS': centroidSizeFeature
  }

  categories = np.concatenate((['haploid'] * len(normHaploidCentroidSize),
                            ['diploid'] * len(normDiploidCentroidSize)))

  dataFrame = pd.DataFrame(data)
  dataFrame.insert(11, 'category', categories)
  print(dataFrame)
  return dataFrame