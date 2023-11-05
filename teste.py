import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn.datasets
from datasets import load_dataset
# Print all the available datasets
from huggingface_hub import list_datasets
from sklearn import datasets, preprocessing
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split

from NormalizationUtils import minMaxNormalization


def plotBars():
  X = ['Group A','Group B','Group C','Group D'] 
  Ygirls = [10,20,20,40] 
  Zboys = [20,30,25,30] 

  X_axis = np.arange(len(X))
  print(X_axis)
  print(X_axis - 0.2)
  print(X_axis + 0.2) 

  plt.bar(X_axis - 0.2, Ygirls, 0.4, label = 'Girls') 
  plt.bar(X_axis + 0.2, Zboys, 0.4, label = 'Boys') 

  plt.xticks(X_axis, X) 
  plt.xlabel("Groups") 
  plt.ylabel("Number of Students") 
  plt.title("Number of Students in each group") 
  plt.legend() 
  plt.show()
# plotBars()



def normalize():
  x_array = np.array([2,3,5,6,7,4,8,7,6])
  normalized_arr = preprocessing.normalize([x_array])
  # normalized_arr = minMaxNormalization(x_array)
  print(normalized_arr[0])

# normalize()


def calculateInterval(haploids, diploids):
  
  plt.scatter(range(0, len(haploids)), haploids)
  plt.scatter(range(0, len(diploids)), diploids)
  plt.legend(["haploid" , "diploid"])
  plt.show()

# calculateInterval([0.556975,0.000000,0.324770,0.113269,0.257528,0.384925,0.973770,0.864172,0.788669,0.345437,0.848710,0.326465,0.165965,0.435375,0.452994,0.515888,1.000000,0.596583,0.798661,0.088981,0.563435,0.496817,0.335340,0.577431,0.868860,0.399696,0.628894,0.415988],
#                   [0.554895,0.672176,0.868067,0.736900,0.800848,0.420240,0.940100,0.628522,0.702037,0.737661,1.000000,0.834109,0.000000,0.122474,0.987839,0.442139,0.450673])


def randomForest():
  X, y = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
  # data["target"] = target
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  # print(data)

  classifier = RandomForestClassifier(n_estimators=1000)
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  positiveClass = 0

  f1TestScoreMean = f1_score(y_test, y_pred, pos_label = positiveClass, average='weighted')
  accuracyTestScoreMean = accuracy_score(y_test, y_pred)
  recallTestScoreMean = recall_score(y_test, y_pred, pos_label = positiveClass, average='weighted')
  precisionTestScoreMean = precision_score(y_test, y_pred, pos_label = positiveClass, average='weighted')

  print(f1TestScoreMean, ": is the f1 score")
  print(accuracyTestScoreMean, ": is the accuracy score")
  print(recallTestScoreMean, ": is the recall score")
  print(precisionTestScoreMean, ": is the precision score")

# print([dataset.id for dataset in list_datasets()])

# # # Load a dataset and print the first example in the training set
# squad_dataset = load_dataset('squad')
# print(squad_dataset['train'][0])

# # Process the dataset - add a column with the length of the context texts
# dataset_with_length = squad_dataset.map(lambda x: {"length": len(x["context"])})

# # Process the dataset - tokenize the context texts (using a tokenizer from the 🤗 Transformers library)
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# tokenized_dataset = squad_dataset.map(lambda x: tokenizer(x['context']), batched=True)


# x = 'looked'
 
# print("Misha %s and %s around"%('walked',x))

# y = np.logspace(0, -9, num = 100)[1:]

# print(len(y))

# plt.plot(y, linestyle = 'dotted')
# plt.show()

# cancer = datasets.load_breast_cancer()
# print("cancer:\n",cancer)
# features = cancer.data
# print("features:\n",features)
# labels = cancer.target
# print("labels:\n",labels)

# clf = AdaBoostClassifier()

# parametros = {'n_estimators':[1, 5, 10],
#               'learning_rate':[0.1, 1, 2]}

# meus_scores = {'accuracy' :make_scorer(accuracy_score),
#                'recall'   :make_scorer(recall_score),
#                'precision':make_scorer(precision_score),
#                'f1'       :make_scorer(f1_score)}

# grid = GridSearchCV(estimator = clf,
#                     param_grid = parametros,
#                     scoring = meus_scores,
#                     refit = 'f1',
#                     cv = 20)

# grid.fit(features, labels)

# df = pd.DataFrame(grid.cv_results_)

# print(df.columns.tolist())
# print(df[['mean_test_f1','mean_test_accuracy','mean_test_recall','mean_test_precision','params']])

# print('best_params:\n', grid.best_params_)
# print('best_params:\n', grid.best_score_)


def error(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return h

def confidence_interval(data, confidence=0.95):
  return st.t.interval(confidence=confidence, df=len(data)-1,
                       loc=np.mean(data),
                       scale=st.sem(data))

def confidenceInterval(data, confidence=0.95):
  return st.norm.interval(confidence, loc=np.mean(data), scale=st.sem(data))

# data = [19.8, 18.5, 17.6, 16.7, 15.8, 15.4, 14.1, 13.6, 11.9, 11.4, 11.4, 8.8, 7.5, 15.4, 15.4, 19.5, 14.9, 12.7, 11.9, 11.4, 10.1, 7.9]
data = [16.8, 17.2, 17.4, 16.9, 16.5, 17.1]
print(error(data, 0.99))
print(confidence_interval(data, 0.99))
print(confidenceInterval(data, 0.99))

f1_ci_0 = []
f1_ci_1 = []
f1_ci_2 = []
f1_ci_3 = []

acc_ci_0 = []
acc_ci_1 = []
acc_ci_2 = []
acc_ci_3 = []

rec_ci_0 = []
rec_ci_1 = []
rec_ci_2 = []
rec_ci_3 = []

prec_ci_0 = []
prec_ci_1 = []
prec_ci_2 = []
prec_ci_3 = []

def appendValues(filename):
  df = pd.read_csv(filename)
  f1_ci_0.append(df.loc[0]['F1'])
  f1_ci_1.append(df.loc[1]['F1'])
  f1_ci_2.append(df.loc[2]['F1'])
  f1_ci_3.append(df.loc[3]['F1'])

  acc_ci_0.append(df.loc[0]['ACC'])
  acc_ci_1.append(df.loc[1]['ACC'])
  acc_ci_2.append(df.loc[2]['ACC'])
  acc_ci_3.append(df.loc[3]['ACC'])

  rec_ci_0.append(df.loc[0]['REC'])
  rec_ci_1.append(df.loc[1]['REC'])
  rec_ci_2.append(df.loc[2]['REC'])
  rec_ci_3.append(df.loc[3]['REC'])

  prec_ci_0.append(df.loc[0]['PREC'])
  prec_ci_1.append(df.loc[1]['PREC'])
  prec_ci_2.append(df.loc[2]['PREC'])
  prec_ci_3.append(df.loc[3]['PREC'])

def main():
  appendValues('out1.csv')
  appendValues('out2.csv')
  appendValues('out3.csv')
  appendValues('out4.csv')
  appendValues('out5.csv')
  appendValues('out6.csv')
  appendValues('out7.csv')
  appendValues('out8.csv')
  appendValues('out9.csv')
  appendValues('out10.csv')

  print(error(f1_ci_0))
  print(error(f1_ci_1))
  print(error(f1_ci_2))
  print(error(f1_ci_3))

  print(error(acc_ci_0))
  print(error(acc_ci_1))
  print(error(acc_ci_2))
  print(error(acc_ci_3))

  print(error(rec_ci_0))
  print(error(rec_ci_1))
  print(error(rec_ci_2))
  print(error(rec_ci_3))

  print(error(prec_ci_0))
  print(error(prec_ci_1))
  print(error(prec_ci_2))
  print(error(prec_ci_3))

# main()