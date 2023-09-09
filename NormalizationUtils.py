import math

import numpy as np


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

  return out

def logNormalization(list):
  out = []

  for i in list:
    out.append(math.log(i, 2))

  return out