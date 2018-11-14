""" Online Passive Agressive class. """

from time import time
import numpy as np
import pandas as pd


class OnlinePassiveAgressive:
    """ Online Passive-Agressive Algorithm. """

    def __init__(self, d, tao=0, C=1):
        self.w = np.zeros((d,))
        self.d = d
        self.C = C

        switch = {
            0: self._tao_classic,
            1: self._tao_first,
            2: self._tao_second
        }
        self.compute_tao = switch[tao]

    def predict(self, x):
        """ prediction for input example x. """
        return np.sign(self.w.dot(x.T))

    def _loss(self, x, y):
        """ compute the loss according to an example and a correct label. """
        return max(0, 1 - y * self.w.dot(x.T))

    def online_learn(self, x, y):
        """ one iteration of online learning at a time. """
        l = self._loss(x, y)
        tao = self.compute_tao(x, l)
        self.w += tao * y * x

    def _tao_classic(self, x, l):
        """ classic update. """
        return l / np.linalg.norm(x) ** 2

    def _tao_first(self, x, l):
        """ first relaxation. """
        return min(self.C, l / np.linalg.norm(x) ** 2)

    def _tao_second(self, x, l):
        """ second relaxation. """
        return l / (np.linalg.norm(x) ** 2 + (1 / 2 * self.C))


if __name__ == '__main__':
    df = pd.read_csv('mushrooms.data', header=None)
    for col in range(df.shape[1]):
        vals = df[col].unique()
        # map numerical values for characters
        mapper = {vals[i]: i+1 for i in range(len(vals))}
        df[col] = df[col].map(mapper)

    # get numpy array
    data = df.values
    np.random.seed(777)
    np.random.shuffle(data)

    labels = data[:, 0] * 2 - 3  # from (1,2) to (+1,-1) classification
    data = data[:, 1:]

    n, d = data.shape

    trainsize = 70 * n // 100  # use 70% of dataset for training
    testsize = n - trainsize

    X_train = data[:trainsize]
    y_train = labels[:trainsize]
    X_test = data[trainsize:]
    y_test = labels[trainsize:]

    clf = OnlinePassiveAgressive(d, tao=2, C=2)

    correct = 0
    start = time()
    for i in range(trainsize):
        yihat = clf.predict(X_train[i])
        correct += int(y_train[i] == yihat)
        clf.online_learn(X_train[i], y_train[i])

    print(f'train time elapsed: {(time() - start) * 100}ms')
    print(f'train online accuracy: {correct * 100 // trainsize}%')

    correct = 0
    start = time()
    for i in range(testsize):
        yihat = clf.predict(X_test[i])
        correct += int(y_test[i] == yihat)
        # do not learn from testing dataset
        #clf.online_learn(X_test[i], y_test[i])

    print(f'test time elapsed: {(time() - start) * 100}ms')
    print(f'test accuracy: {correct * 100 // testsize}%')
