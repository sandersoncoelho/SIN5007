import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from DatasetLoader import loadDataset

DATASET, FEATURE_NAMES = loadDataset()
K = 10
POSITIVE_CLASS = 'diploid'

descriptions = [
  'Todas as\ncaracterísticas (11)',
  'PCA (3)',
  'Branch and Bound (2)',
  'SBS (2)'
]

class Score:
  def __init__(self, accuracy, f1, precision, recall):
    self.accuracy = accuracy
    self.f1 = f1
    self.precision = precision
    self.recall = recall

class EstimatorResult:
  def __init__(self,
               accuracy, f1, precision, recall,
               accuracyError, f1Error, precisionError, recallError):
    self.accuracy = accuracy
    self.f1 = f1
    self.precision = precision
    self.recall = recall

    self.accuracyError = accuracyError
    self.f1Error = f1Error
    self.precisionError = precisionError
    self.recallError = recallError

  def __str__(self):
    return f"\n\nacc:{self.accuracy},\nf1:{self.f1},\nprec:{self.precision},\nrec:{self.recall}\n" \
      f"acc_error:{self.accuracyError},\nf1_error:{self.f1Error},\nprec_error:{self.precisionError},\nrec_error:{self.recallError}"

def error(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return h

def applyStratifiedKFold(X, y, k):
  kf = StratifiedKFold(n_splits = k)
  return list(enumerate(kf.split(X, y)))

def getEstimatorResultValues(estimatorResult, score):
  return {
    "ACC": (estimatorResult.accuracy, estimatorResult.accuracyError),
    "F1": (estimatorResult.f1, estimatorResult.f1Error),
    "PREC": (estimatorResult.precision, estimatorResult.precisionError),
    "REC": (estimatorResult.recall, estimatorResult.recallError)
  }[score]

def plotScore(results, score, title):
  x = descriptions
  x_axis = np.arange(len(x))
  formatPrecision = lambda x : "%.2f" % (x * 100)
  barPosition = [-0.3, 0, 0.3]
  i = 0
  plt.figure(figsize=(10,6))

  for label, estimatorResult in results:
    y, error = getEstimatorResultValues(estimatorResult, score)
    plt.bar(x_axis + barPosition[i], y, width = 0.3, yerr = error, capsize = 3, ecolor = 'black', label = label)

    for j, v in enumerate(y):
      plt.text(j + barPosition[i] - 0.1, v - 0.3, formatPrecision(v), color = 'black')

    i += 1

  plt.xticks(x_axis, x)
  plt.title(title)
  plt.grid(axis = 'y')
  plt.legend()
  plt.show()

def appendResult(estimatorResult, score):
  estimatorResult.accuracy.append(np.mean(score.accuracy))
  estimatorResult.f1.append(np.mean(score.f1))
  estimatorResult.precision.append(np.mean(score.precision))
  estimatorResult.recall.append(np.mean(score.recall))

  estimatorResult.accuracyError.append(error(score.accuracy))
  estimatorResult.f1Error.append(error(score.f1))
  estimatorResult.precisionError.append(error(score.precision))
  estimatorResult.recallError.append(error(score.recall))

def runEstimator(estimator, parameters, features, train, test, score):
  X_train = train[features]
  y_train = train['target']

  X_test = test[features]
  y_test = test['target']

  scores = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score, labels = [POSITIVE_CLASS], average = 'macro', zero_division = 0),
    'precision': make_scorer(precision_score, labels = [POSITIVE_CLASS], average = 'macro', zero_division = 0),
    'recall': make_scorer(recall_score, labels = [POSITIVE_CLASS], average = 'macro', zero_division = 0)
  }

  grid = GridSearchCV(estimator = estimator,
                      param_grid = parameters,
                      scoring = scores,
                      refit = 'recall',
                      cv = K)
  
  grid.fit(X_train, y_train)
  print('best score:', grid.best_score_)
  print('best params: ', grid.best_params_)
  y_pred = grid.predict(X_test)
  
  accuracy = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, labels = [POSITIVE_CLASS], average = 'macro', zero_division = 0)
  precision = precision_score(y_test, y_pred, labels = [POSITIVE_CLASS], average = 'macro', zero_division = 0)
  recall = recall_score(y_test, y_pred, labels = [POSITIVE_CLASS], average = 'macro', zero_division = 0)

  score.accuracy.append(accuracy)
  score.f1.append(f1)
  score.precision.append(precision)
  score.recall.append(recall)

def runNaiveBayes(features, train, test, score):
  parameters = {'var_smoothing': np.logspace(0, -9, num = 100)}
  runEstimator(GaussianNB(), parameters, features, train, test, score)

def runRandomForest(features, train, test, score):
  parameters = {
    'n_estimators': [200, 500],
    'max_features': ['log2', 'sqrt'],
    'max_depth' : [4, 5, 6, 7, 8],
    'criterion' :['gini', 'entropy'],
    'oob_score': [False, True]
  }
  runEstimator(RandomForestClassifier(random_state=42), parameters, features, train, test, score)

def runSVM(features, train, test, score):
  parameters = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['linear', 'poly', 'sigmoid', 'rbf']
  }
  runEstimator(SVC(), parameters, features, train, test, score)

def runMLP(features, train, test, score):
  parameters = {
    'hidden_layer_sizes': [(10, 30, 10), (20,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
  }
  runEstimator(MLPClassifier(max_iter=100), parameters, features, train, test, score)

def main():
  naiveBayesScoreAllFeatures = Score([], [], [], [])
  naiveBayesScorePCA = Score([], [], [], [])
  naiveBayesScoreSel1 = Score([], [], [], [])
  naiveBayesScoreSel2 = Score([], [], [], [])

  randomForestScoreAllFeatures = Score([], [], [], [])
  randomForestScorePCA = Score([], [], [], [])
  randomForestScoreSel1 = Score([], [], [], [])
  randomForestScoreSel2 = Score([], [], [], [])

  svmScoreAllFeatures = Score([], [], [], [])
  svmScorePCA = Score([], [], [], [])
  svmScoreSel1 = Score([], [], [], [])
  svmScoreSel2 = Score([], [], [], [])

  mlpScoreAllFeatures = Score([], [], [], [])
  mlpScorePCA = Score([], [], [], [])
  mlpScoreSel1 = Score([], [], [], [])
  mlpScoreSel2 = Score([], [], [], [])

  for i, (train_index, test_index) in applyStratifiedKFold(DATASET, DATASET["target"], K):
    print("\nRunning fold %d of %d" % (i + 1, K))
    train = DATASET.loc[train_index]
    test = DATASET.loc[test_index]

    print("Evaluate all features")
    runNaiveBayes(FEATURE_NAMES, train, test, naiveBayesScoreAllFeatures)
    # runRandomForest(FEATURE_NAMES, train, test, randomForestScoreAllFeatures)
    runSVM(FEATURE_NAMES, train, test, svmScoreAllFeatures)
    runMLP(FEATURE_NAMES, train, test, mlpScoreAllFeatures)

    print("Evaluate PCA features")
    featuresPCA = ["d410", "d310", "CS", "d23", "d78", "d58", "d810"]
    runNaiveBayes(featuresPCA, train, test, naiveBayesScorePCA)
    # runNaiveBayes(featuresPCA, train, test, randomForestScorePCA)
    runSVM(featuresPCA, train, test, svmScorePCA)
    runMLP(featuresPCA, train, test, mlpScorePCA)

    print("Evaluate selector 1")
    featuresSel1 = ["d36", "d68"]
    runNaiveBayes(featuresSel1, train, test, naiveBayesScoreSel1)
    # runRandomForest(["d36", "d68"], train, test, randomForestScoreSel1)
    runSVM(featuresSel1, train, test, svmScoreSel1)
    runMLP(featuresSel1, train, test, mlpScoreSel1)

    print("Evaluate selector 2")
    featuresSel2 = ["d45", "d810"]
    runNaiveBayes(featuresSel2, train, test, naiveBayesScoreSel2)
    # runRandomForest(["d45", "d810"], train, test, randomForestScoreSel2)
    runSVM(featuresSel2, train, test, svmScoreSel2)
    runMLP(featuresSel2, train, test, mlpScoreSel2)

    
  print("\nCalculating mean and confidence interval")

  naiveBayesResult = EstimatorResult([], [], [], [], [], [], [], [])
  appendResult(naiveBayesResult, naiveBayesScoreAllFeatures)
  appendResult(naiveBayesResult, naiveBayesScorePCA)
  appendResult(naiveBayesResult, naiveBayesScoreSel1)
  appendResult(naiveBayesResult, naiveBayesScoreSel2)
  print("naiveBayesResult:", naiveBayesResult)

  # randomForestResult = EstimatorResult([], [], [], [], [], [], [], [])
  # appendResult(randomForestResult, randomForestScoreAllFeatures)
  # appendResult(randomForestResult, randomForestScorePCA)
  # appendResult(randomForestResult, randomForestScoreSel1)
  # appendResult(randomForestResult, randomForestScoreSel2)
  # print("randomForestResult:", randomForestResult)

  svmResult = EstimatorResult([], [], [], [], [], [], [], [])
  appendResult(svmResult, svmScoreAllFeatures)
  appendResult(svmResult, svmScorePCA)
  appendResult(svmResult, svmScoreSel1)
  appendResult(svmResult, svmScoreSel2)
  print("svmResult:", svmResult)

  mlpResult = EstimatorResult([], [], [], [], [], [], [], [])
  appendResult(mlpResult, mlpScoreAllFeatures)
  appendResult(mlpResult, mlpScorePCA)
  appendResult(mlpResult, mlpScoreSel1)
  appendResult(mlpResult, mlpScoreSel2)
  print("mlpResult:", mlpResult)

  results = [
    ("NB", naiveBayesResult),
    ("SVM", svmResult),
    # ("RF", svmResult),
    ("MLP", mlpResult)
  ]

  plotScore(results, 'ACC', 'Acurácia Média')
  plotScore(results, 'F1', 'F1 Média')
  plotScore(results, 'PREC', 'Precisão Média')
  plotScore(results, 'REC', 'Revocação Média')

main()


# naiveBayesResult = EstimatorResult(
#   [0.6298418972332016, 0.625691699604743, 0.5901185770750988, 0.6227272727272727],
#   [0.5918022871580575, 0.556098101771726, 0.5111890302679777, 0.5565731539474142],
#   [0.6666402793001259, 0.571391468868249, 0.527015332448769, 0.6318650793650794],
#   [0.6909090909090909, 0.6363636363636364, 0.6090909090909091, 0.5463636363636363],
#   [0.12249417255069943, 0.09333407525247162, 0.08580418872586916, 0.09405993446656595],
#   [0.2104416587746679, 0.21607369480847632, 0.2156776264492998, 0.14776114883195202],
#   [0.23691256601238497, 0.25433401964699365, 0.23546626351544622, 0.14087313978457844],
#   [0.281306571598293, 0.2826397855675354, 0.28602778293626346, 0.19268464925124495]
# )
# svmResult = EstimatorResult(
#   [0.5642292490118577, 0.5990118577075098, 0.6122529644268774, 0.6132411067193676],
#   [0.5051497822517313, 0.5429077784960138, 0.5529239024708492, 0.5739975923479761],
#   [0.601695755225167, 0.5845649049093322, 0.5482905982905983, 0.6065585721468074],
#   [0.5372727272727272, 0.6, 0.6363636363636364, 0.5918181818181817],
#   [0.08168334102256651, 0.08389421075526324, 0.10504490982630496, 0.08240550987372913],
#   [0.1540849337498743, 0.17258997813238566, 0.21676359727635844, 0.13393426993633847],
#   [0.17899489864446994, 0.20092719464877143, 0.2504033081119112, 0.12967691455043062],
#   [0.21231652276160162, 0.2550215489923256, 0.27420086821102924, 0.18059957665901202]
# )

# mlpResult = EstimatorResult(
#   [0.5015810276679841, 0.5154150197628458, 0.5065217391304347, 0.5023715415019764],
#   [0.29822037510656435, 0.20013440860215054, 0.11016042780748662, 0.19411764705882356],
#   [0.24328063241106718, 0.15045454545454545, 0.09328063241106718, 0.14347826086956522],
#   [0.40909090909090906, 0.3, 0.14545454545454545, 0.3],
#   [0.0369490948222457, 0.024220579538670253, 0.018577637432593174, 0.015768211897005134],
#   [0.22920550246266572, 0.23096338677311468, 0.16927485103038667, 0.22359131520317338],
#   [0.18583670542494016, 0.17404500521330118, 0.1407337861988029, 0.16526314601973688],
#   [0.33195579743338466, 0.34555021440490435, 0.23786036320934367, 0.3455502144049043]
# )

# results = [
#   ("NB", naiveBayesResult),
#   ("SVM", svmResult),
#   # ("RF", svmResult),
#   ("MLP", mlpResult)
# ]

# plotScore(results, 'ACC', 'Acurácia Média')
# plotScore(results, 'F1', 'F1 Média')
# plotScore(results, 'PREC', 'Precisão Média')
# plotScore(results, 'REC', 'Revocação Média')