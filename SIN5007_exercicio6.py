import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from LoaderFeatures import FEATURE_NAMES, getFeaturesAsDataFrame

DATA_FRAME_TRAIN = getFeaturesAsDataFrame('diploid.json', 'haploid.json')
DATA_FRAME_TEST = getFeaturesAsDataFrame('diploid_test.json', 'haploid_test.json')

description = {
  'ALL': 'Todas as\ncaracterísticas (11)',
  'PCA': 'PCA (3)',
  'SEL1': 'Branch and Bound (2)',
  'SEL2': 'SBS (2)'
}


def runRandomForestForFeatures(features, positiveClass, result):
  X_train = DATA_FRAME_TRAIN[features]
  y_train = DATA_FRAME_TRAIN["target"]

  X_test = DATA_FRAME_TEST[features]
  y_test = DATA_FRAME_TEST["target"]

  classifier = RandomForestClassifier(n_estimators=1000, oob_score = True, max_samples = 180)
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)

  f1TestScoreMean = f1_score(y_test, y_pred, pos_label = positiveClass)
  accuracyTestScoreMean = accuracy_score(y_test, y_pred)
  recallTestScoreMean = recall_score(y_test, y_pred, pos_label = positiveClass)
  precisionTestScoreMean = precision_score(y_test, y_pred, pos_label = positiveClass)

  print(f1TestScoreMean, ": is the f1 score")
  print(accuracyTestScoreMean, ": is the accuracy score")
  print(recallTestScoreMean, ": is the recall score")
  print(precisionTestScoreMean, ": is the precision score")

  result["F1"].append(f1TestScoreMean)
  result["ACC"].append(accuracyTestScoreMean)
  result["REC"].append(recallTestScoreMean)
  result["PREC"].append(precisionTestScoreMean)

  result["F1_CI"].append(0.0)
  result["ACC_CI"].append(0.0)
  result["REC_CI"].append(0.0)
  result["PREC_CI"].append(0.0)

  return result

def plotScore(resultDF, score, ci, title):
  x = resultDF[score].index
  y = resultDF[score].values
  error = resultDF[ci].values

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

def main():
  positiveClass = 'diploid'

  result = {
    "F1": [],
    "F1_CI": [],
    "ACC": [],
    "ACC_CI": [],
    "REC": [],
    "REC_CI": [],
    "PREC": [],
    "PREC_CI": [],
  }

  print("\n\nALL FEATURES")
  result = runRandomForestForFeatures(FEATURE_NAMES, positiveClass, result)

  print("\n\nPCA REATURES")
  pcaFeatures = ["CS", "d68", "d45"]
  result = runRandomForestForFeatures(pcaFeatures, positiveClass, result)

  print("\n\nSELECTOR 1")
  sel1Features = ["d36", "d68"]
  result = runRandomForestForFeatures(sel1Features, positiveClass, result)

  print("\n\nSELECTOR 2")
  sel2Features = ["d45", "d810"]
  result = runRandomForestForFeatures(sel2Features, positiveClass, result)

  resultDF = pd.DataFrame(result, index = [description["ALL"],
                                           description["PCA"],
                                           description["SEL1"],
                                           description["SEL2"]])
  print(resultDF)

  resultDF.xs(description["ALL"])['F1_CI'] = 0.01052918096697295
  resultDF.xs(description["PCA"])['F1_CI'] = 0.027383931145068315
  resultDF.xs(description["SEL1"])['F1_CI'] = 0.006192680828392794
  resultDF.xs(description["SEL2"])['F1_CI'] = 0.011355209259602126
  
  resultDF.xs(description["ALL"])['ACC_CI'] = 0.009023769734074108
  resultDF.xs(description["PCA"])['ACC_CI'] = 0.01253958082668654
  resultDF.xs(description["SEL1"])['ACC_CI'] = 0.01124074931483758
  resultDF.xs(description["SEL2"])['ACC_CI'] = 0.014608169580436576
  
  resultDF.xs(description["ALL"])['REC_CI'] = 0.013306806839652893
  resultDF.xs(description["PCA"])['REC_CI'] = 0.03319300807064083
  resultDF.xs(description["SEL1"])['REC_CI'] = 0.0
  resultDF.xs(description["SEL2"])['REC_CI'] = 0.013306806839652893
  
  resultDF.xs(description["ALL"])['PREC_CI'] = 0.010568268741996652
  resultDF.xs(description["PCA"])['PREC_CI'] = 0.021180946989518112
  resultDF.xs(description["SEL1"])['PREC_CI'] = 0.011049176210686946
  resultDF.xs(description["SEL2"])['PREC_CI'] = 0.012001557379496052

  print(resultDF)

  resultDF.to_csv(index=False, path_or_buf='out10.csv')
  plotScore(resultDF, 'F1', 'F1_CI', 'F1 Média')
  plotScore(resultDF, 'ACC', 'ACC_CI', 'Acurácia Média')
  plotScore(resultDF, 'REC', 'REC_CI', 'Revocação Média')
  plotScore(resultDF, 'PREC', 'PREC_CI', 'Precisão Média')

main()

