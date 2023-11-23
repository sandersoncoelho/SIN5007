import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, normalize

from DatasetLoader import loadDataset

DATASET, FEATURE_NAMES = loadDataset()
print(DATASET)

def main():
  X = DATASET.drop('target', axis=1)
  pca = PCA()
  pca.fit_transform(X)

  pcas = pca.components_.shape[0]
  most_important = [np.abs(pca.components_[i]).argmax() for i in range(pcas)]
  most_important_names = [FEATURE_NAMES[most_important[i]] for i in range(pcas)]
  dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(pcas)}
  df = pd.DataFrame(dic.items())
  print(df)

  print("PCA:")
  for i in pca.explained_variance_ratio_:
    print("{:.8f}".format(float(i)))

main()
