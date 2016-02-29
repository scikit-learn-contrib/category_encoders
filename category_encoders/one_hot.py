"""

.. module:: one_hot
  :synopsis:
  :platform:

"""

from sklearn import preprocessing

__author__ = 'willmcginnis'


def one_hot(X_in):
    """
    One hot encoding transforms the matrix of categorical variables (integers) into a matrix of binary columns
    where each class is represented by its own column. In this case the sklearn implementation is used.

    :param X:
    :return:

    """

    return preprocessing.OneHotEncoder().fit_transform(X_in)