import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

from LoaderFeatures import FEATURE_NAMES


def plotPcaScatterMatrix(pca, components, dataFrame, dimension):
  labels = {
      str(i): f"PC {i+1} ({var:.1f}%)"
      for i, var in enumerate(pca.explained_variance_ratio_ * 100)
  }

  fig = px.scatter_matrix(
      components,
      labels=labels,
      dimensions=range(dimension),
      color=dataFrame["category"]
  )
  fig.update_traces(diagonal_visible=False)
  fig.show()

def plotPca3D(pca, components, dataFrame):
  total_var = pca.explained_variance_ratio_.sum() * 100

  fig = px.scatter_3d(
      components, x=0, y=1, z=2, color=dataFrame['category'],
      title=f'Total Explained Variance: {total_var:.2f}%',
      labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
  )
  fig.show()

def printAllPca(pca):
  pcas= pca.components_.shape[0]
  most_important = [np.abs(pca.components_[i]).argmax() for i in range(pcas)]
  most_important_names = [FEATURE_NAMES[most_important[i]] for i in range(pcas)]
  dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(pcas)}
  df = pd.DataFrame(dic.items())
  print(df)

  print("PCA:")
  for i in pca.explained_variance_ratio_:
    print("{:.8f}".format(float(i)))

  pcaQuantity = len(pca.explained_variance_ratio_)
  plt.bar(range(1, pcaQuantity + 1), pca.explained_variance_ratio_)
  plt.xticks(np.arange(1, pcaQuantity + 1, step=1))
  plt.xlabel("Componentes Principais")
  plt.ylabel("Variância Explicada")
  plt.show()

def runPCA(dataFrame):
  pca = PCA()
  components = pca.fit_transform(dataFrame[FEATURE_NAMES])
  
  plotPcaScatterMatrix(pca, components, dataFrame, 4)
  
  #3D
  # pca = PCA(n_components=3)
  # plotPca3D(pca, components, dataFrame)

  printAllPca(pca)