import copy
from category_encoders import utils
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as encoders
import pandas as pd


class MultiClassWrapper(BaseEstimator, TransformerMixin):
    """Extend supervised encoders to n-class labels, where n >= 2.

    The label can be numerical (e.g.: 0, 1, 2, 3,...,n), string or categorical (pandas.Categorical).
    The label is first encoded into n-1 binary columns. Subsequently, the inner supervised encoder
    is executed for each binarized label.

    The names of the encoded features are suffixed with underscore and the corresponding class name
    (edge scenarios like 'dog'+'cat_frog' vs. 'dog_cat'+'frog' are not currently handled).

    The implementation is experimental and the API may change in the future.
    The order of the returned features may change in the future.


    Parameters
    ----------

    feature_encoder: Object
        an instance of a supervised encoder.


    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> from category_encoders.wrapper import MultiClassWrapper
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> y = (y/10).round().astype(int)  # we create 6 artificial classes
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = TargetEncoder(cols=['CHAS', 'RAD'])
    >>> wrapper = MultiClassWrapper(enc)
    >>> encoded =wrapper.fit_transform(X, y)
    >>> print(encoded.info())
    """

    def __init__(self, feature_encoder):
        self.feature_encoder = feature_encoder
        self.feature_encoders = {}
        self.label_encoder = None

    def fit(self, X, y, **kwargs):
        # unite the input into pandas types
        X = utils.convert_input(X)
        y = utils.convert_input(y)
        y.columns = ['target']

        # apply one-hot-encoder on the label
        self.label_encoder = encoders.OneHotEncoder(handle_missing='error', handle_unknown='error', cols=['target'], drop_invariant=True,
                                                    use_cat_names=True)
        labels = self.label_encoder.fit_transform(y)
        labels.columns = [column[7:] for column in labels.columns]
        labels = labels.iloc[:, 1:]  # drop one label

        # train the feature encoders
        for class_name, label in labels.iteritems():
            self.feature_encoders[class_name] = copy.deepcopy(self.feature_encoder).fit(X, label)

    def transform(self, X):
        # unite the input into pandas types
        X = utils.convert_input(X)

        # initialization
        encoded = None
        feature_encoder = None
        all_new_features = pd.DataFrame()

        # transform the features
        for class_name, feature_encoder in self.feature_encoders.items():
            encoded = feature_encoder.transform(X)

            # decorate the encoded features with the label class suffix
            new_features = encoded[feature_encoder.cols]
            new_features.columns = [str(column) + '_' + class_name for column in new_features.columns]

            all_new_features = pd.concat((all_new_features, new_features), axis=1)

        # add features that were not encoded
        result = pd.concat((encoded[encoded.columns[~encoded.columns.isin(feature_encoder.cols)]], all_new_features), axis=1)

        return result

    def fit_transform(self, X, y=None, **fit_params):
        # When we are training the feature encoders, we have to use fit_transform() method on the features.

        # unite the input into pandas types
        X = utils.convert_input(X)
        y = utils.convert_input(y)
        y.columns = ['target']

        # apply one-hot-encoder on the label
        self.label_encoder = encoders.OneHotEncoder(handle_missing='error', handle_unknown='error', cols=['target'], drop_invariant=True,
                                                    use_cat_names=True)
        labels = self.label_encoder.fit_transform(y)
        labels.columns = [column[7:] for column in labels.columns]
        labels = labels.iloc[:, 1:]  # drop one label

        # initialization of the feature encoders
        encoded = None
        feature_encoder = None
        all_new_features = pd.DataFrame()

        # fit_transform the feature encoders
        for class_name, label in labels.iteritems():
            feature_encoder = copy.deepcopy(self.feature_encoder)
            encoded = feature_encoder.fit_transform(X, label)

            # decorate the encoded features with the label class suffix
            new_features = encoded[feature_encoder.cols]
            new_features.columns = [str(column) + '_' + class_name for column in new_features.columns]

            all_new_features = pd.concat((all_new_features, new_features), axis=1)
            self.feature_encoders[class_name] = feature_encoder

        # add features that were not encoded
        result = pd.concat((encoded[encoded.columns[~encoded.columns.isin(feature_encoder.cols)]], all_new_features), axis=1)

        return result
