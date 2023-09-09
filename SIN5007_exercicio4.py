import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import LoaderFeatures as loader
from NormalizationUtils import minMaxNormalization

feature_names = ['d12', 'd13', 'd34', 'd36', 'd45', 'd67', 'd68', 'd79', 'd810', 'd910', 'CS']

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
  most_important_names = [feature_names[most_important[i]] for i in range(pcas)]
  dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(pcas)}
  df = pd.DataFrame(dic.items())
  print(df)

  print("PCA:")
  for i in pca.explained_variance_ratio_:
    print("{:.8f}".format(float(i)))

def runPCA(dataFrame):
  pca = PCA()
  components = pca.fit_transform(dataFrame[feature_names])
  
  plotPcaScatterMatrix(pca, components, dataFrame, 4)
  
  #3D
  # pca = PCA(n_components=3)
  # plotPca3D(pca, components, dataFrame)

  printAllPca(pca)

def runNaiveBayes(dataFrame):
  haploidDF = dataFrame.loc[dataFrame['category'] == 'haploid']
  diploidDF = dataFrame.loc[dataFrame['category'] == 'diploid']

  haploidTrain, haploidTest = train_test_split(haploidDF, test_size=0.2)
  diploidTrain, diploidTest = train_test_split(diploidDF, test_size=0.2)
  print('train:', haploidTrain)
  print('test:', haploidTest)
  print('train:', diploidTrain)
  print('test:', diploidTest)

def main():
  dataFrame = getFeaturesAsDataFrame()
 
  # runPCA(dataFrame)

  runNaiveBayes(dataFrame)

main()