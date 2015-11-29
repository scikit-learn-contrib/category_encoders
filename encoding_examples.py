import copy
import random
import time
import pandas as pd
import numpy as np
from patsy.highlevel import dmatrix
from sklearn import cross_validation, naive_bayes, preprocessing

__author__ = 'willmcginnis'


def get_cars_data():
    """
    Load the cars dataset, split it into X and y, and then call the label encoder to get an integer y column.

    :return:
    """

    df = pd.read_csv('source_data/cars/car.data.txt')
    X = df.reindex(columns=[x for x in df.columns.values if x != 'class'])
    y = df.reindex(columns=['class'])
    y = preprocessing.LabelEncoder().fit_transform(y.values.reshape(-1, ))

    mapping = [
        {'col': 'buying', 'mapping': [('vhigh', 0), ('high', 1), ('med', 2), ('low', 3)]},
        {'col': 'maint', 'mapping': [('vhigh', 0), ('high', 1), ('med', 2), ('low', 3)]},
        {'col': 'doors', 'mapping': [('2', 0), ('3', 1), ('4', 2), ('5more', 3)]},
        {'col': 'persons', 'mapping': [('2', 0), ('4', 1), ('more', 2)]},
        {'col': 'lug_boot', 'mapping': [('small', 0), ('med', 1), ('big', 2)]},
        {'col': 'safety', 'mapping': [('high', 0), ('med', 1), ('low', 2)]},
    ]

    return X, y, mapping


def get_mushroom_data():
    """
    Load the mushroom dataset, split it into X and y, and then call the label encoder to get an integer y column.

    :return:
    """

    df = pd.read_csv('source_data/mushrooms/agaricus-lepiota.csv')
    X = df.reindex(columns=[x for x in df.columns.values if x != 'class'])
    y = df.reindex(columns=['class'])
    y = preprocessing.LabelEncoder().fit_transform(y.values.reshape(-1, ))

    # this data is truly categorical, with no known concept of ordering
    mapping = None

    return X, y, mapping


def get_splice_data():
    """
    Load the mushroom dataset, split it into X and y, and then call the label encoder to get an integer y column.

    :return:
    """

    df = pd.read_csv('source_data/splice/splice.csv')
    X = df.reindex(columns=[x for x in df.columns.values if x != 'class'])
    X['dna'] = X['dna'].map(lambda x: list(str(x).strip()))
    for idx in range(60):
        X['dna_%d' % (idx, )] = X['dna'].map(lambda x: x[idx])
    del X['dna']

    y = df.reindex(columns=['class'])
    y = preprocessing.LabelEncoder().fit_transform(y.values.reshape(-1, ))

    # this data is truly categorical, with no known concept of ordering
    mapping = None

    return X, y, mapping


def score_models(clf, X, y, runs=10):
    """
    Takes in a classifier that supports multiclass classification, and X and a y, and returns a cross validation score.

    :param clf:
    :param X:
    :param y:
    :return:
    """

    scores = []

    for _ in range(runs):
        scores.append(cross_validation.cross_val_score(clf, X, y, n_jobs=-1))

    return float(np.mean(scores))


def ordinal_encoding(X_in, mapping=None):
    """
    Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
    in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
    are assumed to have no true order and integers are selected at random.

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    if mapping is not None:
        for switch in mapping:
            for category in switch.get('mapping'):
                X.loc[X[switch.get('col')] == category[0], switch.get('col')] = str(category[1])
            X[switch.get('col')] = X[switch.get('col')].astype(int).reshape(-1, )
    else:
        for col in X.columns.values:
            categories = list(set(X[col].values))
            random.shuffle(categories)
            for idx, val in enumerate(categories):
                X.loc[X[col] == val, col] = str(idx)
            X[col] = X[col].astype(int).reshape(-1, )

    return X


def one_hot(X_in):
    """
    One hot encoding transforms the matrix of categorical variables (integers) into a matrix of binary columns
    where each class is represented by its own column. In this case the sklearn implementation is used.

    :param X:
    :return:

    """

    return preprocessing.OneHotEncoder().fit_transform(X_in)


def binary(X_in):
    """
    Binary encoding encodes the integers as binary code with one column per digit.

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    bin_cols = []
    for col in X.columns.values:
        # figure out how many digits we need to represent the classes present
        if X[col].max() == 0:
            digits = 1
        else:
            digits = int(np.ceil(np.log2(X[col].max())))

        # map the ordinal column into a list of these digits, of length digits
        X[col] = X[col].map(lambda x: list("{0:b}".format(int(x)))) \
            .map(lambda x: x if len(x) == digits else [0 for _ in range(digits - len(x))] + x)

        for dig in range(digits):
            X[col + '_%d' % (dig, )] = X[col].map(lambda x: int(x[dig]))
            bin_cols.append(col + '_%d' % (dig, ))

    X = X.reindex(columns=bin_cols)

    return X


def sum_coding(X_in):
    """

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    bin_cols = []
    for col in X.columns.values:
        mod = dmatrix("C(%s, Sum)" % (col, ), X)
        for dig in range(len(mod[0])):
            X[col + '_%d' % (dig, )] = mod[:, dig]
            bin_cols.append(col + '_%d' % (dig, ))

    X = X.reindex(columns=bin_cols)

    return X


def polynomial_coding(X_in):
    """

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    bin_cols = []
    for col in X.columns.values:
        mod = dmatrix("C(%s, Poly)" % (col, ), X)
        for dig in range(len(mod[0])):
            X[col + '_%d' % (dig, )] = mod[:, dig]
            bin_cols.append(col + '_%d' % (dig, ))

    X = X.reindex(columns=bin_cols)

    return X


def helmert_coding(X_in):
    """

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    bin_cols = []
    for col in X.columns.values:
        mod = dmatrix("C(%s, Helmert)" % (col, ), X)
        for dig in range(len(mod[0])):
            X[col + '_%d' % (dig, )] = mod[:, dig]
            bin_cols.append(col + '_%d' % (dig, ))

    X = X.reindex(columns=bin_cols)

    return X


def backward_difference_coding(X_in):
    """

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    bin_cols = []
    for col in X.columns.values:
        mod = dmatrix("C(%s, Diff)" % (col, ), X)
        for dig in range(len(mod[0])):
            X[col + '_%d' % (dig, )] = mod[:, dig]
            bin_cols.append(col + '_%d' % (dig, ))

    X = X.reindex(columns=bin_cols)
    X.fillna(0.0)
    return X


def main():
    """
    Here we iterate through the datasets and score them with a classifier using different encodings.
    :return:
    """

    scores = []
    for gen in [('Mushroom', get_mushroom_data), ('Cars', get_cars_data), ('Splice', get_splice_data)]:
        # first get the dataset
        X, y, mapping = gen[1]()

        # create a simple classifier for evaluating encodings (we use bernoulli naive bayed because the data will all be
        # boolean (1/0) by the time the classifier sees it.
        clf = naive_bayes.BernoulliNB()

        # we have two base cases, informed and uninformed ordinal source_data.  So first we form those datasets, and
        # score with these two datasets, then use them to create more.
        start_time = time.time()
        X_ordinal = ordinal_encoding(X, mapping)
        score_uninformed_ordinal = score_models(clf, X_ordinal, y)
        scores.append(['Ordinal', gen[0], X_ordinal.shape[1], score_uninformed_ordinal, time.time() - start_time])

        # then try one hot encoding
        start_time = time.time()
        X_onehot = one_hot(X_ordinal)
        score_onehot = score_models(clf, X_onehot, y)
        scores.append(['One-Hot Encoded', gen[0], X_onehot.shape[1], score_onehot, time.time() - start_time])

        # finally try binary encoding with uninformed basis
        start_time = time.time()
        X_binary2 = binary(X_ordinal)
        score_binary2 = score_models(clf, X_binary2, y)
        scores.append(['Binary Encoded', gen[0], X_binary2.shape[1], score_binary2, time.time() - start_time])

        # now lets try some implimentations from statsmodels-patsy
        # source: (http://statsmodels.sourceforge.net/devel/contrasts.html)

        # sum encoding with uninformed basis
        start_time = time.time()
        X_sum_coded = sum_coding(X_ordinal)
        score_sum_coded = score_models(clf, X_sum_coded, y)
        scores.append(['Sum Coding', gen[0], X_sum_coded.shape[1], score_sum_coded, time.time() - start_time])

        if gen[0] != 'Splice':
            # polynomial encoding with uninformed basis
            start_time = time.time()
            X_poly_coded = polynomial_coding(X_ordinal)
            score_poly_coded = score_models(clf, X_poly_coded, y)
            scores.append(['Polynomial Coding', gen[0], X_poly_coded.shape[1], score_poly_coded, time.time() - start_time])

            # backward difference encoding with uninformed basis (extremely slow)
            start_time = time.time()
            X_diff_coded = backward_difference_coding(X_ordinal)
            score_diff_coded = score_models(clf, X_diff_coded, y)
            scores.append(['Backward Difference Coding', gen[0], X_diff_coded.shape[1], score_diff_coded, time.time() - start_time])

            # helmert encoding with uninformed basis
            start_time = time.time()
            X_helmert_coded = helmert_coding(X_ordinal)
            score_helmert_coded = score_models(clf, X_helmert_coded, y)
            scores.append(['Helmert Coding', gen[0], X_helmert_coded.shape[1], score_helmert_coded, time.time() - start_time])

    results = pd.DataFrame(scores, columns=['Encoding', 'Dataset', 'Dimensionality', 'Avg. Score', 'Elapsed Time'])
    print(results.sort_values(by=['Dataset', 'Avg. Score']))

    cars = results[results['Dataset'] == 'Cars'].sort_values(by=['Avg. Score']).to_html()
    print(cars)

    mushrooms = results[results['Dataset'] == 'Mushroom'].sort_values(by=['Avg. Score']).to_html()
    print(mushrooms)

    splice = results[results['Dataset'] == 'Splice'].sort_values(by=['Avg. Score']).to_html()
    print(splice)


if __name__ == '__main__':
    main()