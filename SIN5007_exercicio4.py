import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from LoaderFeatures import FEATURE_NAMES, getFeaturesAsDataFrame


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

def splitTrainTest(dataFrame):
  haploidDF = dataFrame.loc[dataFrame['category'] == 'haploid']
  diploidDF = dataFrame.loc[dataFrame['category'] == 'diploid']

  haploidTrain, haploidTest = train_test_split(haploidDF, test_size=0.2)
  diploidTrain, diploidTest = train_test_split(diploidDF, test_size=0.2)
  train = pd.concat([haploidTrain, diploidTrain])
  test = pd.concat([haploidTest, diploidTest])

  print('haploidTrain:', len(haploidTrain))
  print('haploidTest:', len(haploidTest))
  print('diploidTrain:', len(diploidTrain))
  print('diploidTest:', len(diploidTest))
  print('Train:', len(train))
  print('Test:', len(test))
  return train, test

def runNaiveBayes(dataFrame, features):
  train, test = splitTrainTest(dataFrame)  

  x_train = train[features]
  y_train = train['category']

  x_test = test[features]
  y_test = test['category']

  model = GaussianNB()
  model.fit(x_train, y_train)

  acc_train = model.score(x_train, y_train)
  print("\nAccuracy on train data = %0.4f " % acc_train)
  acc_test = model.score(x_test, y_test)
  print("Accuracy on test data =  %0.4f " % acc_test)

  y_predicteds = model.predict(x_test)

  cm = confusion_matrix(y_test, y_predicteds)
  ax= plt.subplot()
  sns.heatmap(cm, annot=True, ax=ax)
  ax.xaxis.set_ticklabels(['diploid', 'haploid'])
  ax.yaxis.set_ticklabels(['diploid', 'haploid'])
  plt.show()

def main():
  dataFrame = getFeaturesAsDataFrame()
 
  runPCA(dataFrame)

  # runNaiveBayes(dataFrame, feature_names)
  
  # runNaiveBayes(dataFrame, ['CS', 'd68', 'd45'])

main()