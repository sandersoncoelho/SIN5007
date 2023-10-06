import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Classifiers import applyStratifiedKFold, runGridSearchCV, runNaiveBayes
from LoaderFeatures import FEATURE_NAMES, getFeaturesAsDataFrame

DATA_FRAME_TRAIN = getFeaturesAsDataFrame('diploid.json', 'haploid.json')
DATA_FRAME_TEST = getFeaturesAsDataFrame('diploid_test.json', 'haploid_test.json')

description = {
  'ALL': 'Todas as\ncaracterísticas (11)',
  'PCA': 'PCA (3)',
  'SEL1': 'Branch and Bound (2)',
  'SEL2': 'SBS (2)'
}

def runNaiveBayesForFeatures(k, features):
  f1ScoreMean = 0; accuracyScoreMean = 0; recallScoreMean = 0; precisionScoreMean = 0
  
  for i, (train_index, test_index) in applyStratifiedKFold(DATA_FRAME_TRAIN, DATA_FRAME_TRAIN["category"], k):
    print('\n', i)
    train = DATA_FRAME_TRAIN.loc[train_index]
    test = DATA_FRAME_TRAIN.loc[test_index]
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

def runNaiveBayesWithGridSearchCV(k, features, result):
  X_train = DATA_FRAME_TRAIN[features]
  y_train = DATA_FRAME_TRAIN["category"]

  X_test = DATA_FRAME_TEST[features]
  y_test = DATA_FRAME_TEST["category"]

  f1ScoreMean, f1Std,\
    accuracyScoreMean, accStd,\
      recallScoreMean, recStd,\
        precisionScoreMean, preStd, \
          f1TestScoreMean, accuracyTestScoreMean, \
            recallTestScoreMean, precisionTestScoreMean \
              = runGridSearchCV(X_train, y_train, X_test, y_test, 'diploid', k, 'recall')
  
  result["F1"].append(f1ScoreMean)
  result["F1_STD"].append(f1Std)

  result["ACC"].append(accuracyScoreMean)
  result["ACC_STD"].append(accStd)

  result["REC"].append(recallScoreMean)
  result["REC_STD"].append(recStd)

  result["PREC"].append(precisionScoreMean)
  result["PREC_STD"].append(preStd)

  result["F1_TEST"].append(f1TestScoreMean)
  result["ACC_TEST"].append(accuracyTestScoreMean)
  result["REC_TEST"].append(recallTestScoreMean)
  result["PREC_TEST"].append(precisionTestScoreMean)

  return result

def plotScore(resultDF, score, std, title):
  x = resultDF[score].index
  y = resultDF[score].values
  error = resultDF[std].values

  formatPrecision = lambda x : "%.4f" % x
  plt.bar(x, y, yerr = error, capsize = 3, ecolor = 'black')
  for i, v in enumerate(y):
    plt.text(i - 0.2, v - 0.3, formatPrecision(v),
            color = 'black')
    plt.text(i + 0.03, v, formatPrecision(error[i]),
            color = 'black')

  plt.title(title)
  plt.grid(axis = 'y')
  plt.show()

def plotScoreWithouStd(resultDF, score, title):
  x = resultDF[score].index
  y = resultDF[score].values

  formatPrecision = lambda x : "%.4f" % x
  plt.bar(x, y, capsize = 3, ecolor = 'black')
  for i, v in enumerate(y):
    plt.text(i - 0.2, v - 0.2, formatPrecision(v),
            color = 'black')

  plt.title(title)
  plt.grid(axis = 'y')
  plt.show()

def main():
  K = 10

  result = {
    "F1": [],
    "F1_STD": [],
    "ACC": [],
    "ACC_STD": [],
    "REC": [],
    "REC_STD": [],
    "PREC": [],
    "PREC_STD": [],

    "F1_TEST": [],
    "ACC_TEST": [],
    "REC_TEST": [],
    "PREC_TEST": []
  }

  # ALL FEATURES
  print("\n\nALL FEATURES")
  # f1ScoreMean, accuracyScoreMean, recallScoreMean, precisionScoreMean = runNaiveBayesForFeatures(K, FEATURE_NAMES)
  result = runNaiveBayesWithGridSearchCV(K, FEATURE_NAMES, result)

  # PCA FEATURES
  print("\n\nPCA REATURES")
  pcaFeatures = ["CS", "d68", "d45"]
  # f1ScoreMean, accuracyScoreMean, recallScoreMean, precisionScoreMean = runNaiveBayesForFeatures(K, pcaFeatures)
  result = runNaiveBayesWithGridSearchCV(K, pcaFeatures, result)

  # SELECTOR 1
  print("\n\nSELECTOR 1")
  sel1Features = ["d36", "d68"]
  # f1ScoreMean, accuracyScoreMean, recallScoreMean, precisionScoreMean = runNaiveBayesForFeatures(K, sel1Features)
  result = runNaiveBayesWithGridSearchCV(K, sel1Features, result)

  #SELECTOR 2
  print("\n\nSELECTOR 2")
  sel2Features = ["d45", "d810"]
  result = runNaiveBayesWithGridSearchCV(K, sel2Features, result)

  resultDF = pd.DataFrame(result, index = [description["ALL"],
                                           description["PCA"],
                                           description["SEL1"],
                                           description["SEL2"]])
  print(resultDF)

  plotScore(resultDF, 'F1', 'F1_STD', 'F1 Média')
  plotScore(resultDF, 'ACC', 'ACC_STD', 'Acurácia Média')
  plotScore(resultDF, 'REC', 'REC_STD', 'Revocação Média')
  plotScore(resultDF, 'PREC', 'PREC_STD', 'Precisão Média')

  plotScoreWithouStd(resultDF, 'F1_TEST', 'F1 Média - Test')
  plotScoreWithouStd(resultDF, 'ACC_TEST', 'Acurácia Média - Test')
  plotScoreWithouStd(resultDF, 'REC_TEST', 'Revocação Média - Test')
  plotScoreWithouStd(resultDF, 'PREC_TEST', 'Precisão Média - Test')
  
main()
