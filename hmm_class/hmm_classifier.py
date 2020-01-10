# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Demonstrate how HMMs can be used for classification.
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

from sqlalchemy import create_engine

import string
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hmmc_scaled_concat import HMM
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from nltk import pos_tag, word_tokenize

class HMMClassifier:
    def __init__(self):
        pass

    def fit(self, X, Y, V):
        K = len(set(Y)) # number of classes - assume 0..K-1
        N = len(Y)
        self.models = []
        self.priors = []
        for k in range(K):
            # gather all the training data for this class
            thisX = [x for x, y in zip(X, Y) if y == k]
            C = len(thisX)
            self.priors.append(np.log(C) - np.log(N))

            hmm = HMM(3, 2)
            hmm.fit(thisX, max_iter=80)
            self.models.append(hmm)

    def score(self, X, Y):
        N = len(Y)
        correct = 0
        for x, y in zip(X, Y):
            lls = [hmm.log_likelihood(x) + prior for hmm, prior in zip(self.models, self.priors)]
            p = np.argmax(lls)
            if p == y:
                correct += 1
        return float(correct) / N


# def remove_punctuation(s):
#     return s.translate(None, string.punctuation)

def get_tags(s):
    tuples = pos_tag(word_tokenize(s))
    return [y for x, y in tuples]

def get_data():
    word2idx = {}
    current_idx = 0
    X = []
    Y = []
    for fn, label in zip(('robert_frost.txt', 'edgar_allan_poe.txt'), (0, 1)):
        count = 0
        for line in open(fn):
            line = line.rstrip()
            if line:
                print(line)
                # tokens = remove_punctuation(line.lower()).split()
                tokens = get_tags(line)
                if len(tokens) > 1:
                    # scan doesn't work nice here, technically could fix...
                    for token in tokens:
                        if token not in word2idx:
                            word2idx[token] = current_idx
                            current_idx += 1
                    sequence = np.array([word2idx[w] for w in tokens])
                    X.append(sequence)
                    Y.append(label)
                    count += 1
                    print(count)
                    if count >= 50:
                        break
    print("Vocabulary:", word2idx.keys())
    return X, Y, current_idx

def get_stock_data():
    engine = create_engine('mysql+pymysql://crawler:4%&e3H6J63VVPk#d@62.234.184.151:3306/crawler?charset=utf8mb4')
    df = pd.read_sql('SELECT TRADE_DT as Date,S_DQ_ADJCLOSE/S_DQ_ADJFACTOR as Close,S_DQ_VOLUME/1000 as Volume, S_DQ_HIGH, S_DQ_LOW FROM wind.ASHAREEODPRICES WHERE S_INFO_WINDCODE="601318.SH" AND TRADE_DT >= "20190601" ORDER BY TRADE_DT',
    engine,
        columns=[
            "Date", "Close", "Volume", "S_DQ_HIGH", "S_DQ_LOW"
        ],
        index_col="Date", parse_dates={'Date': '%Y%m%d'}
    )
    df["Returns"] = df["Close"].pct_change()
    df["Vola"] = df["S_DQ_HIGH"]/df["S_DQ_LOW"] - 1
    df["Returns5"] = df["Close"].pct_change(periods=2)
    X = []
    Y = []
    for key,val in df[10:].iterrows():
        y = 1 if val['Returns5'] > 0 else 0
        returns = np.array(df[:key][-10:-2]['Returns'])
        vols = np.array(df[:key][-10:-2]['Volume'])
        volas = np.array(df[:key][-10:-2]['Vola'])
        X.append(np.column_stack([scale(returns), scale(vols), scale(volas)]))
        Y.append(y)
    return X, Y, len(Y)

def main():
    X, Y, V = get_stock_data()
    print("len(X):", len(X))
    print("Vocabulary size:", V)
    X, Y = shuffle(X, Y)
    N = 20 # number to test
    Xtrain, Ytrain = X[:-N], Y[:-N]
    Xtest, Ytest = X[-N:], Y[-N:]

    model = HMMClassifier()
    model.fit(Xtrain, Ytrain, V)
    print("Score:", model.score(Xtest, Ytest))


if __name__ == '__main__':
    main()
