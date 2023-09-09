import json
import math
import os

import numpy as np

import CentroidUtils as centroidUtils


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
    d79Feature = np.append(d79Feature, math.dist(p7, p8))
    d810Feature = np.append(d810Feature, math.dist(p8, p10))
    d910Feature = np.append(d910Feature, math.dist(p9, p10))
    centroidSizeFeature = np.append(centroidSizeFeature, centroidSize)

  return d12Feature, d13Feature, d34Feature, d36Feature, d45Feature, \
    d67Feature, d68Feature, d79Feature, d810Feature, d910Feature, \
    centroidSizeFeature