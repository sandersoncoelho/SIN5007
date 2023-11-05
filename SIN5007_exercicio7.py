import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from DatasetLoader import FEATURE_NAMES, loadDataset

DATASET = loadDataset()
print(DATASET)
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
    'f1': make_scorer(f1_score, labels = [POSITIVE_CLASS], average = 'macro'),
    'precision': make_scorer(precision_score, labels = [POSITIVE_CLASS], average = 'macro', zero_division = 0),
    'recall': make_scorer(recall_score, labels = [POSITIVE_CLASS], average = 'macro')
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
  f1 = f1_score(y_test, y_pred, labels = [POSITIVE_CLASS], average = 'macro')
  precision = precision_score(y_test, y_pred, labels = [POSITIVE_CLASS], average = 'macro', zero_division = 0)
  recall = recall_score(y_test, y_pred, labels = [POSITIVE_CLASS], average = 'macro')

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
    'kernel': ['rbf','linear']
  }
  runEstimator(SVC(), parameters, features, train, test, score)

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

  for i, (train_index, test_index) in applyStratifiedKFold(DATASET, DATASET["target"], K):
    print("\nRunning fold %d of %d" % (i + 1, K))
    train = DATASET.loc[train_index]
    test = DATASET.loc[test_index]

    print("Evaluate all features")
    runNaiveBayes(FEATURE_NAMES, train, test, naiveBayesScoreAllFeatures)
    runRandomForest(FEATURE_NAMES, train, test, randomForestScoreAllFeatures)
    runSVM(FEATURE_NAMES, train, test, svmScoreAllFeatures)

    print("Evaluate PCA features")
    runNaiveBayes(["CS", "d68", "d45"], train, test, naiveBayesScorePCA)
    runNaiveBayes(["CS", "d68", "d45"], train, test, randomForestScorePCA)
    runSVM(["CS", "d68", "d45"], train, test, svmScorePCA)

    print("Evaluate selector 1")
    runNaiveBayes(["d36", "d68"], train, test, naiveBayesScoreSel1)
    runRandomForest(["d36", "d68"], train, test, randomForestScoreSel1)
    runSVM(["d36", "d68"], train, test, svmScoreSel1)

    print("Evaluate selector 2")
    runNaiveBayes(["d45", "d810"], train, test, naiveBayesScoreSel2)
    runRandomForest(["d45", "d810"], train, test, randomForestScoreSel2)
    runSVM(["d45", "d810"], train, test, svmScoreSel2)

    
  print("\nCalculating mean and confidence interval")

  naiveBayesResult = EstimatorResult([], [], [], [], [], [], [], [])
  appendResult(naiveBayesResult, naiveBayesScoreAllFeatures)
  appendResult(naiveBayesResult, naiveBayesScorePCA)
  appendResult(naiveBayesResult, naiveBayesScoreSel1)
  appendResult(naiveBayesResult, naiveBayesScoreSel2)
  print("naiveBayesResult:", naiveBayesResult)

  randomForestResult = EstimatorResult([], [], [], [], [], [], [], [])
  appendResult(randomForestResult, randomForestScoreAllFeatures)
  appendResult(randomForestResult, randomForestScorePCA)
  appendResult(randomForestResult, randomForestScoreSel1)
  appendResult(randomForestResult, randomForestScoreSel2)
  print("randomForestResult:", randomForestResult)

  svmResult = EstimatorResult([], [], [], [], [], [], [], [])
  appendResult(svmResult, svmScoreAllFeatures)
  appendResult(svmResult, svmScorePCA)
  appendResult(svmResult, svmScoreSel1)
  appendResult(svmResult, svmScoreSel2)
  print("svmResult:", svmResult)

  results = [
    ("NB", naiveBayesResult),
    ("SVM", svmResult),
    ("RF", randomForestResult)
  ]

  plotScore(results, 'ACC', 'Acurácia Média')
  plotScore(results, 'F1', 'F1 Média')
  plotScore(results, 'PREC', 'Precisão Média')
  plotScore(results, 'REC', 'Revocação Média')

main()
