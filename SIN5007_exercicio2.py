import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def calculateCentroid(points):
  lenght = len(points)
  
  if (lenght == 0): return None

  x = y = 0

  for p in points:
    x += p[0]
    y += p[1]

  return (x / lenght, y / lenght)

def plotLandmarks(filename, points, centroid):
  x = []; y = []; n = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
  centroidSize = 0

  for p in points:
    # print(p)
    x.append(p[0])
    y.append(p[1])
    centroidSize += math.pow(math.dist(p, centroid), 2)
  
  centroidSize = math.sqrt(centroidSize)
  # print('centroidSize:', centroidSize)

  # plt.scatter(x, y)
  # for i, txt in enumerate(n):
  #   plt.annotate(txt, (x[i], y[i]))
  
  # plt.scatter(centroid[0], centroid[1], color='red')

  # plt.title(filename)
  # plt.gca().invert_yaxis()
  # plt.show()
  return centroidSize

def getFeatures(annotationFile):
  annotations = json.load(open(os.path.join('./', annotationFile)))
  image_metadata = annotations['_via_img_metadata']
  image_id_list =  annotations['_via_image_id_list']

  d12Feature = np.array([])
  d13Feature = np.array([])
  d34Feature = np.array([])
  d45Feature = np.array([])
  d36Feature = np.array([])
  d67Feature = np.array([])
  d68Feature = np.array([])
  d79Feature = np.array([])
  d810Feature = np.array([])
  d910Feature = np.array([])
  centroidFeature = np.array([])
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
    centroid = calculateCentroid([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
    centroidSize = plotLandmarks(image_metadata[keyFilename]['filename'], [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10], centroid)

    d12Feature = np.append(d12Feature, math.dist(p1, p2))
    d13Feature = np.append(d13Feature, math.dist(p1, p3))
    d34Feature = np.append(d34Feature, math.dist(p3, p4))
    d45Feature = np.append(d45Feature, math.dist(p4, p5))
    d36Feature = np.append(d36Feature, math.dist(p3, p6))
    d67Feature = np.append(d67Feature, math.dist(p6, p7))
    d68Feature = np.append(d68Feature, math.dist(p6, p8))
    d79Feature = np.append(d79Feature, math.dist(p7, p8))
    d810Feature = np.append(d810Feature, math.dist(p8, p10))
    d910Feature = np.append(d910Feature, math.dist(p9, p10))
    centroidFeature = np.append(centroidFeature, centroid)
    centroidSizeFeature = np.append(centroidSizeFeature, centroidSize)

  return d12Feature, d13Feature, d34Feature, d45Feature, d36Feature, \
    d67Feature, d68Feature, d79Feature, d810Feature, d910Feature, \
    centroidFeature, centroidSizeFeature


haploidFeatureD12, haploidFeatureD13, haploidFeatureD34, haploidFeatureD45, haploidFeatureD36, \
  haploidFeatureD67, haploidFeatureD68, haploidFeatureD79, haploidFeatureD810, haploidFeatureD910, \
    haploidCentroid, haploidCentroidSize = getFeatures('haploid.json')

diploidFeatureD12, diploidFeatureD13, diploidFeatureD34, diploidFeatureD45, diploidFeatureD36, \
  diploidFeatureD67, diploidFeatureD68, diploidFeatureD79, diploidFeatureD810, diploidFeatureD910, \
    diploidCentroid, diploidCentroidSize = getFeatures('diploid.json')

# plt.scatter(np.arange(start = 1, stop = len(haploidCentroidSize) + 1), haploidCentroidSize, label = 'haploid')
# plt.scatter(np.arange(start = 1, stop = len(diploidCentroidSize) + 1), diploidCentroidSize, label = 'diploid')
# plt.title("Centroid Size")
# plt.xlabel("Instâncias")
# plt.ylabel("CS")
# plt.legend()
# plt.show()
'''
plt.subplot(1, 2, 1)
plt.scatter(np.arange(start = 1, stop = len(haploidFeatureD12) + 1), haploidFeatureD12, label = 'haploid')
plt.scatter(np.arange(start = 1, stop = len(diploidFeatureD12) + 1), diploidFeatureD12, label = 'diploid')
plt.title("Distancia entre os landmarks 1 a 2")
plt.xlabel("Individuos")
plt.ylabel("Pixels")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(np.arange(start = 1, stop = len(haploidFeatureD13) + 1), haploidFeatureD13, label = 'haploid')
plt.scatter(np.arange(start = 1, stop = len(diploidFeatureD13) + 1), diploidFeatureD13, label = 'diploid')
plt.title("Distancia entre os landmarks 1 a 3")
plt.xlabel("Individuos")
plt.ylabel("Pixels")
plt.legend()

plt.tight_layout()
plt.show()

plt.subplot(1, 2, 1)
plt.scatter(np.arange(start = 1, stop = len(haploidFeatureD34) + 1), haploidFeatureD34, label = 'haploid')
plt.scatter(np.arange(start = 1, stop = len(diploidFeatureD34) + 1), diploidFeatureD34, label = 'diploid')
plt.title("Distancia entre os landmarks 3 a 4")
plt.xlabel("Individuos")
plt.ylabel("Pixels")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(np.arange(start = 1, stop = len(haploidFeatureD36) + 1), haploidFeatureD36, label = 'haploid')
plt.scatter(np.arange(start = 1, stop = len(diploidFeatureD36) + 1), diploidFeatureD36, label = 'diploid')
plt.xlabel("Individuos")
plt.ylabel("Pixels")
plt.title("Distancia entre os landmarks 3 a 6")
plt.legend()

plt.tight_layout()
plt.show()

plt.subplot(1, 2, 1)
plt.scatter(np.arange(start = 1, stop = len(haploidFeatureD45) + 1), haploidFeatureD45, label = 'haploid')
plt.scatter(np.arange(start = 1, stop = len(diploidFeatureD45) + 1), diploidFeatureD45, label = 'diploid')
plt.title("Distancia entre os landmarks 4 a 5")
plt.xlabel("Individuos")
plt.ylabel("Pixels")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(np.arange(start = 1, stop = len(haploidFeatureD67) + 1), haploidFeatureD67, label = 'haploid')
plt.scatter(np.arange(start = 1, stop = len(diploidFeatureD67) + 1), diploidFeatureD67, label = 'diploid')
plt.xlabel("Individuos")
plt.ylabel("Pixels")
plt.title("Distancia entre os landmarks 6 a 7")
plt.legend()

plt.tight_layout()
plt.show()

plt.subplot(1, 2, 1)
plt.scatter(np.arange(start = 1, stop = len(haploidFeatureD68) + 1), haploidFeatureD68, label = 'haploid')
plt.scatter(np.arange(start = 1, stop = len(diploidFeatureD68) + 1), diploidFeatureD68, label = 'diploid')
plt.xlabel("Individuos")
plt.ylabel("Pixels")
plt.title("Distancia entre os landmarks 6 a 8")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(np.arange(start = 1, stop = len(haploidFeatureD79) + 1), haploidFeatureD79, label = 'haploid')
plt.scatter(np.arange(start = 1, stop = len(diploidFeatureD79) + 1), diploidFeatureD79, label = 'diploid')
plt.xlabel("Individuos")
plt.ylabel("Pixels")
plt.title("Distancia entre os landmarks 7 a 9")
plt.legend()

plt.tight_layout()
plt.show()

plt.subplot(1, 2, 1)
plt.scatter(np.arange(start = 1, stop = len(haploidFeatureD810) + 1), haploidFeatureD810, label = 'haploid')
plt.scatter(np.arange(start = 1, stop = len(diploidFeatureD810) + 1), diploidFeatureD810, label = 'diploid')
plt.xlabel("Individuos")
plt.ylabel("Pixels")
plt.title("Distancia entre os landmarks 8 a 10")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(np.arange(start = 1, stop = len(haploidFeatureD910) + 1), haploidFeatureD910, label = 'haploid')
plt.scatter(np.arange(start = 1, stop = len(diploidFeatureD910) + 1), diploidFeatureD910, label = 'diploid')
plt.xlabel("Individuos")
plt.ylabel("Pixels")
plt.title("Distancia entre os landmarks 9 a 10")
plt.legend()

plt.tight_layout()
plt.show()
'''

featureD12 = np.concatenate((haploidFeatureD12, diploidFeatureD12))
featureD13 = np.concatenate((haploidFeatureD13, diploidFeatureD13))
featureD34 = np.concatenate((haploidFeatureD34, diploidFeatureD34))
featureD36 = np.concatenate((haploidFeatureD36, diploidFeatureD36))
featureD45 = np.concatenate((haploidFeatureD45, diploidFeatureD45))
featureD67 = np.concatenate((haploidFeatureD67, diploidFeatureD67))
featureD68 = np.concatenate((haploidFeatureD68, diploidFeatureD68))
featureD79 = np.concatenate((haploidFeatureD79, diploidFeatureD79))
featureD810 = np.concatenate((haploidFeatureD810, diploidFeatureD810))
featureD910 = np.concatenate((haploidFeatureD910, diploidFeatureD910))
centroidSizeFeature = np.concatenate((haploidCentroidSize, diploidCentroidSize))

data = []

for i in range(len(featureD12)):
  data.append([featureD12[i], featureD13[i], featureD34[i], featureD36[i], featureD45[i], \
               featureD67[i], featureD68[i], featureD79[i], featureD810[i], featureD910[i], centroidSizeFeature[i]])

import pandas as pd

feature_names = ['d12', 'd13', 'd34', 'd36', 'd45', 'd67', 'd68', 'd79', 'd810', 'd910', 'CS']

pca = PCA()
pca.fit(data)

pcas= pca.components_.shape[0]
most_important = [np.abs(pca.components_[i]).argmax() for i in range(pcas)]
most_important_names = [feature_names[most_important[i]] for i in range(pcas)]
dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(pcas)}
df = pd.DataFrame(dic.items())
print(df)

print("PCA:")
for i in pca.explained_variance_ratio_:
  print("{:.8f}".format(float(i)))
