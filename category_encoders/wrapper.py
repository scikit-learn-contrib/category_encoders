import copy
from category_encoders import utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
import category_encoders as encoders
import pandas as pd
import numpy as np


class PolynomialWrapper(BaseEstimator, TransformerMixin):
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
    >>> from category_encoders.wrapper import PolynomialWrapper
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> y = (y/10).round().astype(int)  # we create 6 artificial classes
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = TargetEncoder(cols=['CHAS', 'RAD'])
    >>> wrapper = PolynomialWrapper(enc)
    >>> encoded = wrapper.fit_transform(X, y)
    >>> print(encoded.info())
    """

    def __init__(self, feature_encoder):
        self.feature_encoder = feature_encoder
        self.feature_encoders = {}
        self.label_encoder = None

    def fit(self, X, y, **kwargs):
        # unite the input into pandas types
        X, y = utils.convert_inputs(X, y)
        y = pd.DataFrame(y, columns=['target'])

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
        X, y = utils.convert_inputs(X, y)
        y = pd.DataFrame(y, columns=['target'])

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


class NestedCVWrapper(BaseEstimator, TransformerMixin):
    """
    Extends supervised encoders with the nested cross validation on the training data to minimise overfitting.

    For a validation or a test set, supervised encoders can be used as follows:

        X_train_encoded = encoder.fit_transform(X_train, y_train)
        X_valid_encoded = encoder.transform(X_valid)

    However, the downstream model will be overfitting to the encoded training data due to target leakage.
    Using out-of-fold encodings is an effective way to prevent target leakage. This is equivalent to:

        X_train_encoded = np.zeros(X.shape)
        for trn, val in kfold.split(X, y):
            encoder.fit(X[trn], y[trn])
            X_train_encoded[val] = encoder.transform(X[val])

    This can be used in place of the "inner folds" as discussed here:
        https://sebastianraschka.com/faq/docs/evaluate-a-model.html

    See README.md for a list of supervised encoders.

    Discussion: Although leave-one-out encoder internally performs leave-one-out cross-validation, it is
    actually the most overfitting supervised model in our library. To illustrate the issue, let's imagine we
    have a totally unpredictive nominal feature and a perfectly balanced binary label. A supervised encoder
    should encode the feature into a constant vector as the feature is unpredictive of the label. But when we
    use leave-one-out cross-validation, the label ratio cease to be perfectly balanced and the wrong class
    label always becomes the majority in the training fold. Leave-one-out encoder returns a seemingly
    predictive feature. And the downstream model starts to overfit to the encoded feature. Unfortunately,
    even 10-fold cross-validation is not immune to this effect:
        http://www.kdd.org/exploration_files/v12-02-4-UR-Perlich.pdf
    To decrease the effect, it is recommended to use a low count of the folds. And that is the reason why
    this wrapper uses 5 folds by default.

    Based on the empirical results, only LeaveOneOutEncoder benefits greatly from this wrapper. The remaining
    encoders can be used without this wrapper.


    Parameters
    ----------
    feature_encoder: Object
        an instance of a supervised encoder.

    cv: int or sklearn cv Object
        if an int is given, StratifiedKFold is used by default, where the int is the number of folds.

    shuffle: boolean, optional
        whether to shuffle each classes samples before splitting into batches. Ignored if a CV method is provided.

    random_state: int, RandomState instance or None, optional, default=None
        if int, random_state is the seed used by the random number generator. Ignored if a CV method is provided.


    Example
    -------
    >>> from category_encoders import *
    >>> from category_encoders.wrapper import NestedCVWrapper
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import GroupKFold, train_test_split
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> # we create 6 artificial classes and a train/validation/test split
    >>> y = (y/10).round().astype(int)
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> X_train, X_test, y_train, _ = train_test_split(X, y)
    >>> X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train)
    >>> # Define the nested CV encoder for a supervised encoder
    >>> enc_nested = NestedCVWrapper(TargetEncoder(cols=['CHAS', 'RAD']), random_state=42)
    >>> # Encode the X data for train, valid & test
    >>> X_train_enc, X_valid_enc, X_test_enc = enc_nested.fit_transform(X_train, y_train, X_test=(X_valid, X_test))
    >>> print(X_train_enc.info())
    """

    def __init__(self, feature_encoder, cv=5, shuffle=True, random_state=None):
        self.feature_encoder = feature_encoder
        self.__name__ = feature_encoder.__class__.__name__
        self.shuffle = shuffle
        self.random_state = random_state

        if type(cv) == int:
            self.cv = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
        else:
            self.cv = cv

    def fit(self, X, y, **kwargs):
        """
        Calls fit on the base feature_encoder without nested cross validation
        """
        self.feature_encoder.fit(X, y, **kwargs)

    def transform(self, X):
        """
        Calls transform on the base feature_encoder without nested cross validation
        """
        return self.feature_encoder.transform(X)

    def fit_transform(self, X, y=None, X_test=None, groups=None, **fit_params):
        """
        Creates unbiased encodings from a supervised encoder as well as infer encodings on a test set
        :param X: array-like, shape = [n_samples, n_features]
                  Training vectors for the supervised encoder, where n_samples is the number of samples
                  and n_features is the number of features.
        :param y: array-like, shape = [n_samples]
                  Target values for the supervised encoder.
        :param X_test, optional: array-like, shape = [m_samples, n_features] or a tuple of array-likes (X_test, X_valid...)
                       Vectors to be used for inference by an encoder (e.g. test or validation sets) trained on the
                       full X & y sets. No nested folds are used here
        :param groups: Groups to be passed to the cv method, e.g. for GroupKFold
        :param fit_params:
        :return: array, shape = [n_samples, n_numeric + N]
                 Transformed values with encoding applied. Returns multiple arrays if X_test is not None
        """
        X, y = utils.convert_inputs(X, y)

        # Get out-of-fold encoding for the training data
        out_of_fold = np.zeros(X.shape)

        for trn_idx, oof_idx in self.cv.split(X, y, groups):
            feature_encoder = copy.deepcopy(self.feature_encoder)
            feature_encoder.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            out_of_fold[oof_idx] = feature_encoder.transform(X.iloc[oof_idx])

        out_of_fold = pd.DataFrame(out_of_fold, columns=X.columns)

        # Train the encoder on all the training data for testing data
        self.feature_encoder = copy.deepcopy(self.feature_encoder)
        self.feature_encoder.fit(X, y)

        if X_test is None:
            return out_of_fold
        else:
            if type(X_test) == tuple:
                encoded_data = (out_of_fold, )
                for dataset in X_test:
                    encoded_data = encoded_data + (self.feature_encoder.transform(dataset), )
                return encoded_data
            else:
                return out_of_fold, self.feature_encoder.transform(X_test)
