import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB


def applyHoldout(dataFrame):
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

def applyStratifiedKFold(X, y, k):
  kf = StratifiedKFold(n_splits = k)
  return list(enumerate(kf.split(X, y)))

def runNaiveBayes(train, test, features, plotCM = False):
  X_train = train[features]
  y_train = train['category']

  X_test = test[features]
  y_test = test['category']

  model = GaussianNB()
  model.fit(X_train, y_train)

  y_predicteds = model.predict(X_test)

  f1Score = f1_score(y_test, y_predicteds, pos_label='diploid')
  accuracyScore = accuracy_score(y_test, y_predicteds);
  recallScore = recall_score(y_test, y_predicteds, pos_label='diploid')
  precisionScore = precision_score(y_test, y_predicteds, pos_label='diploid')

  if plotCM:
    cm = confusion_matrix(y_test, y_predicteds)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)
    ax.xaxis.set_ticklabels(['diploid', 'haploid'])
    ax.yaxis.set_ticklabels(['diploid', 'haploid'])
    plt.show()

  return (f1Score, accuracyScore, recallScore, precisionScore)

def runGridSearchCV(X_train, y_train, X_test, y_test, positiveClass, k, measure):
  parameters = {'var_smoothing': np.logspace(0, -9, num = 100)}

  scores = {
    'accuracy': make_scorer(accuracy_score),
    'recall': make_scorer(recall_score, labels = [positiveClass], average = 'macro'),
    'precision': make_scorer(precision_score, labels = [positiveClass], average = 'macro'),
    'f1': make_scorer(f1_score, labels = [positiveClass], average = 'macro')
  }

  grid = GridSearchCV(estimator = GaussianNB(),
                      param_grid = parameters,
                      scoring = scores,
                      refit = measure,
                      cv = k)

  grid.fit(X_train, y_train)

  df = pd.DataFrame(grid.cv_results_)
  
  f1Std = np.std(df['mean_test_f1'])
  accStd = np.std(df['mean_test_accuracy'])
  recStd = np.std(df['mean_test_recall'])
  precStd = np.std(df['mean_test_precision'])

  df = df[df['rank_test_%s' % measure] == 1]

  print(df[['mean_test_f1', 'std_test_f1',
            'mean_test_accuracy', 'std_test_accuracy',
            'mean_test_recall', 'std_test_recall',
            'mean_test_precision', 'std_test_precision',
            'params']])

  f1ScoreMean = df.iloc[0]['mean_test_f1']
  accuracyScoreMean = df.iloc[0]['mean_test_accuracy']
  recallScoreMean = df.iloc[0]['mean_test_recall']
  precisionScoreMean = df.iloc[0]['mean_test_precision']
  params = df.iloc[0]['params']
  
  print("f1ScoreMean:", f1ScoreMean, ' ', f1Std)
  print("accuracyScoreMean:", accuracyScoreMean, ' ', accStd)
  print("recallScoreMean:", recallScoreMean, ' ', recStd)
  print("precisionScoreMean:", precisionScoreMean, ' ', precStd)

  print('best score:', grid.best_score_)
  print('best params: ', params)

  print("PREDICTS:\n")

  y_pred = grid.predict(X_test)
  f1TestScoreMean = f1_score(y_test, y_pred, labels = [positiveClass], average = 'macro')
  accuracyTestScoreMean = accuracy_score(y_test, y_pred)
  recallTestScoreMean = recall_score(y_test, y_pred, labels = [positiveClass], average = 'macro')
  precisionTestScoreMean = precision_score(y_test, y_pred, labels = [positiveClass], average = 'macro')

  print(f1TestScoreMean, ": is the f1 score")
  print(accuracyTestScoreMean, ": is the accuracy score")
  print(recallTestScoreMean, ": is the recall score")
  print(precisionTestScoreMean, ": is the precision score")
  # print('Results:\n', pd.DataFrame(grid.cv_results_)[['mean_test_recall', 'std_test_recall', 'rank_test_recall']])

  return (f1ScoreMean, f1Std,
          accuracyScoreMean, accStd,
          recallScoreMean, recStd,
          precisionScoreMean, precStd,
          f1TestScoreMean, accuracyTestScoreMean, recallTestScoreMean, precisionTestScoreMean)
