import pandas as pd
from sklearn import preprocessing

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