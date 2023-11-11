import matplotlib.pyplot as plt
import numpy as np

from DatasetLoader import FEATURE_NAMES, loadDataset

DATASET = loadDataset()
print(DATASET)

def plotFeatureByFeature():
  # ['d12', 'd13', 'd34', 'd36', 'd45', 'd67', 'd68', 'd79', 'd810', 'd910', 'CS']
  
  cs = DATASET['CS'].values
  d68 = DATASET['d68'].values
  d45 = DATASET['d45'].values
  d36 = DATASET['d36'].values
  d810 = DATASET['d810'].values
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
  
  axs[1, 3].scatter(d36, d810, c=targetColors, s=10)
  axs[1, 3].set_title("d36 x d810")
  
  fig.tight_layout()
  plt.show()

plotFeatureByFeature()