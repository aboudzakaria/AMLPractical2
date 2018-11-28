""" TP learning from unbalanced data. """

import numpy as np
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

X_unbalanced = np.genfromtxt("train.csv", delimiter=',')
y_unbalanced = np.genfromtxt("train_labels.csv")

X, y = shuffle(X_unbalanced, y_unbalanced, random_state=27)

print(f"count labels ['0': {sum(y_unbalanced==0)}, '1': {sum(y_unbalanced==1)}]")

n = X.shape[0]
d = X.shape[1]
train_size = n * 70 // 100

# split train test
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

print(f"train count labels ['0': {sum(y_train==0)}, '1': {sum(y_train==1)}]")
print(f"test count labels ['0': {sum(y_test==0)}, '1': {sum(y_test==1)}]")


ratio = 0.01
rus = RandomUnderSampler(sampling_strategy=ratio, random_state=27)
X_train, y_train = rus.fit_resample(X_train, y_train.ravel())

print("after Random Under Sampling")
print(f"count labels ['0': {sum(y_train==0)}, '1': {sum(y_train==1)}]")

sm = SMOTE(random_state=27)
X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

print("after SMOTE")
print(f"count labels ['0': {sum(y_train==0)}, '1': {sum(y_train==1)}]")

print('learn XGBoost')
clf = XGBClassifier()
clf.fit(X_train, y_train)

print("predict on splitted set:")
y_pred = clf.predict(X_test)
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
print(f"precision={precision}, recall={recall}, fscore={fscore}, support={support}")

print('predict on test dataset..')
X_unlabeled = np.genfromtxt("test.csv", delimiter=',')
y_pred = clf.predict(X_unlabeled)
print('dump predictions..')
np.savetxt("test_pred_labels.csv", y_pred, delimiter=",")
print('done.')