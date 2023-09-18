import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB

# nb_classifier = GaussianNB()

# params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
# gs_NB = GridSearchCV(estimator=nb_classifier, 
#                  param_grid=params_NB, 
#                  cv=10,
#                  verbose=1, 
#                  scoring='accuracy') 
# gs_NB.fit(x_train, y_train)

# gs_NB.best_params_


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

  # acc_train = model.score(x_train, y_train)
  # print("\nAccuracy on train data = %0.4f " % acc_train)
  # acc_test = model.score(x_test, y_test)
  # print("Accuracy on test data =  %0.4f " % acc_test)

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