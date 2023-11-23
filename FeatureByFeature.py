import math

import matplotlib.pyplot as plt
import numpy as np

from DatasetLoader import loadDataset

DATASET, FEATURE_NAMES = loadDataset()
print(DATASET)

def plotFeatureByFeature():
  # ['d12', 'd13', 'd34', 'd36', 'd45', 'd67', 'd68', 'd79', 'd810', 'd910', 'CS']
  
  cs = DATASET['CS'].values
  d68 = DATASET['d68'].values
  d45 = DATASET['d45'].values
  d36 = DATASET['d36'].values
  d810 = DATASET['d810'].values
  d910 = DATASET['d910'].values
  d210 = DATASET['d210'].values
  target = DATASET['target'].values
  targetColors = ['green' if y == 'diploid' else 'red' for y in target]

  
  fig, axs = plt.subplots(2, 4, figsize=(10,6))
  # axs.figure(figsize=(10,6))
  
  axs[0, 0].scatter(cs, d68, c=targetColors, s=10)
  axs[0, 0].set_title("CS x d68")
  
  axs[1, 0].scatter(cs, d45, c=targetColors, s=10)
  axs[1, 0].set_title("CS x d45")
  
  axs[0, 1].scatter(cs, d36, c=targetColors, s=10)
  axs[0, 1].set_title("CS x d36")
  
  axs[1, 1].scatter(cs, d810, c=targetColors, s=10)
  axs[1, 1].set_title("CS x d810")


  axs[0, 2].scatter(d68, d45, c=targetColors, s=10)
  axs[0, 2].set_title("d68 x d45")
  
  axs[1, 2].scatter(d36, d68, c=targetColors, s=10)
  axs[1, 2].set_title("d36 x d68")
  
  axs[0, 3].scatter(d45, d810, c=targetColors, s=10)
  axs[0, 3].set_title("d45 x d810")
  
  axs[1, 3].scatter(d910, d210, c=targetColors, s=10)
  axs[1, 3].set_title("d910 x d210")
  
  fig.tight_layout()
  plt.show()

# plotFeatureByFeature()

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def getAngle2(a, b, c):
  _a = np.array(a)
  _b = np.array(b)
  _c = np.array(c)

  ba = _a - _b
  bc = _c - _b

  cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
  angle = np.arccos(cosine_angle)

  return np.degrees(angle)
 
print(getAngle((8, 7), (0, 0), (4, 5)))
print(getAngle((4, 5), (0, 0), (8, 7)))
print(getAngle2((8, 7), (0, 0), (4, 5)))
print(getAngle2((4, 5), (0, 0), (8, 7)))