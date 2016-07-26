from memory_profiler import profile
import gc
import category_encoders as ce
from examples.source_data.loaders import get_mushroom_data, get_cars_data, get_splice_data

__author__ = 'willmcginnis'


@profile(precision=4)
def hashing():
    X, _, _ = get_mushroom_data()
    print(X.info())
    enc = ce.HashingEncoder()
    enc.fit(X, None)
    out = enc.transform(X)
    print(out.info())
    del enc, _, X, out


@profile(precision=4)
def onehot():
    X, _, _ = get_mushroom_data()
    print(X.info())
    enc = ce.OneHotEncoder()
    enc.fit(X, None)
    out = enc.transform(X)
    print(out.info())
    del enc, _, X, out


@profile(precision=4)
def ordinal():
    X, _, _ = get_mushroom_data()
    print(X.info())
    enc = ce.OrdinalEncoder()
    enc.fit(X, None)
    out = enc.transform(X)
    print(out.info())
    del enc, _, X, out


@profile(precision=4)
def backward_difference():
    X, _, _ = get_mushroom_data()
    print(X.info())
    enc = ce.BackwardDifferenceEncoder()
    enc.fit(X, None)
    out = enc.transform(X)
    print(out.info())
    del enc, _, X, out


@profile(precision=4)
def binary():
    X, _, _ = get_mushroom_data()
    print(X.info())
    enc = ce.BinaryEncoder()
    enc.fit(X, None)
    out = enc.transform(X)
    print(out.info())
    del enc, _, X, out


@profile(precision=4)
def helmert():
    X, _, _ = get_mushroom_data()
    print(X.info())
    enc = ce.HelmertEncoder()
    enc.fit(X, None)
    out = enc.transform(X)
    print(out.info())
    del enc, _, X, out


@profile(precision=4)
def polynomial():
    X, _, _ = get_mushroom_data()
    print(X.info())
    enc = ce.PolynomialEncoder()
    enc.fit(X, None)
    out = enc.transform(X)
    print(out.info())
    del enc, _, X, out


@profile(precision=4)
def sum_coding():
    X, _, _ = get_mushroom_data()
    print(X.info())
    enc = ce.SumEncoder()
    enc.fit(X, None)
    out = enc.transform(X)
    print(out.info())
    del enc, _, X, out


@profile(precision=4)
def control():
    X, _, _ = get_mushroom_data()
    del X

if __name__ == '__main__':
    gc.collect()
    onehot()
    gc.collect()