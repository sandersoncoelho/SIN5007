import math

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, normalize


def zscoreNormalization(list):
  out = []
  m = np.mean(list)
  sd = np.std(list)

  for i in list:
    out.append((i - m) / sd)
  
  return out

def minMaxNormalization(list):
  out = []
  mininum = min(list)
  maximum = max(list)

  for i in list:
    out.append((i - mininum) / (maximum - mininum))

  # return out
  # normalized_arr = preprocessing.MinMaxScaler().fit(list)
  # normalized_arr = minMaxNormalization(x_array)
  # return np.array(normalized_arr[0])

  return normalize([list], norm='max')[0]

def logNormalization(list):
  out = []

  for i in list:
    out.append(math.log(i, 2))

  return out

def main():
  list = [5.842593,  6.107766,  6.674157,  7.649783,  6.070095,  7.867738,  6.215879,  6.340619,  7.826492,  6.186569,  8.871056]
  print('teste0:\n', list)
  print('teste1:\n', minMaxNormalization(list))
  scaler = MinMaxScaler()
  scaler.fit([list])
  print('teste2:\n', normalize([list], norm='max'))

# main()