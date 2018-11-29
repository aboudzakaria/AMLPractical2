
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.ensemble import BalancedBaggingClassifier # doctest: +NORMALIZE_WHITESPACE

bbc = BalancedBaggingClassifier(random_state=42)

X = np.genfromtxt("train.csv", delimiter=',')
Y = np.genfromtxt("train_labels.csv")

n = X.shape[0]
d = X.shape[1]
train_size = n * 70 // 100

X, Y = shuffle(X, Y, random_state=27)

# split train test
X_train = X[:train_size]
Y_train = Y[:train_size]
X_test = X[train_size:]
Y_test = Y[train_size:]

bbc.fit(X_train, Y_train) 

Y__ = bbc.predict(X_test)

print(confusion_matrix(Y_test, Y__))
coco = precision_recall_fscore_support(Y_test,Y__)

print(coco)


X_final = np.genfromtxt("test.csv", delimiter=',')

Y__ = bbc.predict(X_test)

np.savetxt("test_pred_labels_2.csv", Y__, delimiter=",")

