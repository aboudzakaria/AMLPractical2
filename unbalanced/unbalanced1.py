""" TP learning from unbalanced data. """

import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_supports
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

X_unbalanced = np.genfromtxt("train.csv", delimiter=',')
y_unbalanced = np.genfromtxt("train_labels.csv")

print(f"count labels ['0': {sum(y_unbalanced==0)}, '1': {sum(y_unbalanced==1)}]")

ratio = 4.267e-3
rus = RandomUnderSampler(sampling_strategy=ratio, random_state=27)
X_unsam, y_unsam = rus.fit_resample(X_unbalanced, y_unbalanced.ravel())

print("after Random Under Sampling")
print(f"count labels ['0': {sum(y_unsam==0)}, '1': {sum(y_unsam==1)}]")


sm = SMOTE(random_state=27)
X, y = sm.fit_sample(X_unsam, y_unsam.ravel())

print("after SMOTE")
print(f"count labels ['0': {sum(y==0)}, '1': {sum(y==1)}]")


n = X.shape[0]
d = X.shape[1]
train_size = n * 70 // 100

X, y = shuffle(X, y, random_state=27)

# split train test
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

knn = KNeighborsClassifier(algorithm='brute', n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f"classifier accuracy: {(y_pred == y_test).mean() * 100}%")

#y_pred = knn.predict(X_unbalanced)
#print(f"accuracy on unbalanced (original) data: {(y_pred == y_unbalanced).mean() * 100}%")

X_unlabeled = np.genfromtxt("test.csv", delimiter=',')
km = KMeans(n_clusters=2, random_state=27).fit(X_unlabeled)
y_clust = km.labels_
y_clust_swapped = -1 * y_clust + 1
y_pred = knn.predict(X_unlabeled)
accuracy = max((y_pred == y_clust).mean() * 100, (y_pred == y_clust_swapped).mean() * 100)
print(f"accuracy on unlabeled (clustered) data: {accuracy}%")

np.savetxt("test_clust_labels.csv", y_clust, delimiter=",")
np.savetxt("test_pred_labels.csv", y_pred, delimiter=",")
