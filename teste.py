import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV

x = 'looked'
 
print("Misha %s and %s around"%('walked',x))

y = np.logspace(0, -9, num = 100)[1:]

print(len(y))

plt.plot(y, linestyle = 'dotted')
plt.show()

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