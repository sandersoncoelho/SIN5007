import math

import numpy as np


def calculateCentroid(points):
  lenght = len(points)
  
  if (lenght == 0): return None

  centroid = np.zeros(len(points[0]))

  for p in points:
    for i in range(len(p)):
      centroid[i] += p[i]

  for i in range(len(centroid)):
      centroid[i] /= lenght

  return centroid

def calculateCentroidSize(points, centroid):
  centroidSize = 0

  for p in points:
    centroidSize += math.pow(math.dist(p, centroid), 2)
  
  centroidSize = math.sqrt(centroidSize)
  return centroidSize