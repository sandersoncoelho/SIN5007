import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Classifiers import applyStratifiedKFold, runNaiveBayes
from LoaderFeatures import FEATURE_NAMES, getFeaturesAsDataFrame

dataFrame = getFeaturesAsDataFrame()

description = {
  'ALL': 'Todas as características (11)',
  'PCA': 'PCA (3)',
  'SEL1': 'Branch and Bound (2)'
}

def runNaiveBayesForFeatures(k, features):
  f1ScoreMean = 0; accuracyScoreMean = 0; recallScoreMean = 0; precisionScoreMean = 0
  
  for i, (train_index, test_index) in applyStratifiedKFold(dataFrame, dataFrame["category"], k):
    print('\n', i)
    train = dataFrame.loc[train_index]
    test = dataFrame.loc[test_index]
    print('train:', train)
    print('test:', test)

    f1Score, accuracyScore, recallScore, precisionScore = runNaiveBayes(train, test, features)

    f1ScoreMean += f1Score
    accuracyScoreMean += accuracyScore
    recallScoreMean += recallScore
    precisionScoreMean += precisionScore

    print('\nF1 score:%0.4f' % f1Score)
    print('Accuracy score:%0.4f' % accuracyScore)
    print('Recall score:%0.4f' % recallScore)
    print('Precision score:%0.4f' % precisionScore)

  f1ScoreMean /= k
  accuracyScoreMean /= k
  recallScoreMean /= k
  precisionScoreMean /= k

  return f1ScoreMean, accuracyScoreMean, recallScoreMean, precisionScoreMean

def plotScore(resultDF, score, title):
  x = resultDF[score].index
  y = resultDF[score].values
  error = [0.8, 0.4, 0.2]

  formatPrecision = lambda x : "%.4f" % x
  plt.bar(x, y, yerr = error, capsize = 3, ecolor = '#afafaf')
  for i, v in enumerate(y):
    plt.text(i - 0.2, v + 0.03, formatPrecision(v),
            color = 'gray')

  plt.title(title)
  plt.grid(axis = 'y')
  plt.show()

def main():
  K = 10

  result = {
    "F1": [],
    "ACC": [],
    "REC": [],
    "PREC": []
  }

  # ALL FEATURES
  f1ScoreMean, accuracyScoreMean, recallScoreMean, precisionScoreMean = runNaiveBayesForFeatures(K, FEATURE_NAMES)
  
  result["F1"].append(f1ScoreMean)
  result["ACC"].append(accuracyScoreMean)
  result["REC"].append(recallScoreMean)
  result["PREC"].append(precisionScoreMean)

  # PCA FEATURES
  pcaFeatures = ["CS", "d68", "d45"]
  f1ScoreMean, accuracyScoreMean, recallScoreMean, precisionScoreMean = runNaiveBayesForFeatures(K, pcaFeatures)

  result["F1"].append(f1ScoreMean)
  result["ACC"].append(accuracyScoreMean)
  result["REC"].append(recallScoreMean)
  result["PREC"].append(precisionScoreMean)

  # SELECTOR 1
  sel1Features = ["d36", "d68"]
  f1ScoreMean, accuracyScoreMean, recallScoreMean, precisionScoreMean = runNaiveBayesForFeatures(K, sel1Features)

  result["F1"].append(f1ScoreMean)
  result["ACC"].append(accuracyScoreMean)
  result["REC"].append(recallScoreMean)
  result["PREC"].append(precisionScoreMean)

  resultDF = pd.DataFrame(result, index = [description["ALL"], description["PCA"], description["SEL1"]])
  print(resultDF)

  plotScore(resultDF, 'F1', 'F1 Score Média')
  plotScore(resultDF, 'ACC', 'Accuracy Score Média')
  plotScore(resultDF, 'REC', 'Recall Score Média')
  plotScore(resultDF, 'PREC', 'Precision Score Média')
  
main()
