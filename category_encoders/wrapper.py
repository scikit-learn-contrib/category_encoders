import copy
from category_encoders import utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
import category_encoders as encoders
import pandas as pd
from typing import Dict, Optional


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
    >>> from sklearn.datasets import fetch_openml
    >>> from category_encoders.wrapper import PolynomialWrapper
    >>> display_cols = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
    >>> bunch = fetch_openml(name="house_prices", as_frame=True)
    >>> # need more than one column
    >>> y = bunch.target.map(lambda x: int(min([x, 300000])/50000))
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = TargetEncoder(cols=['CentralAir', 'Heating'])
    >>> wrapper = PolynomialWrapper(enc)
    >>> encoded = wrapper.fit_transform(X, y)
    >>> print(encoded.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 17 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   Id            1460 non-null   float64
     1   MSSubClass    1460 non-null   float64
     2   MSZoning      1460 non-null   object 
     3   LotFrontage   1201 non-null   float64
     4   YearBuilt     1460 non-null   float64
     5   CentralAir_3  1460 non-null   float64
     6   Heating_3     1460 non-null   float64
     7   CentralAir_2  1460 non-null   float64
     8   Heating_2     1460 non-null   float64
     9   CentralAir_5  1460 non-null   float64
     10  Heating_5     1460 non-null   float64
     11  CentralAir_6  1460 non-null   float64
     12  Heating_6     1460 non-null   float64
     13  CentralAir_1  1460 non-null   float64
     14  Heating_1     1460 non-null   float64
     15  CentralAir_0  1460 non-null   float64
     16  Heating_0     1460 non-null   float64
    dtypes: float64(16), object(1)
    memory usage: 194.0+ KB
    None
    """

    def __init__(self, feature_encoder: utils.BaseEncoder):
        self.feature_encoder: utils.BaseEncoder = feature_encoder
        self.feature_encoders: Dict[str, utils.BaseEncoder] = {}
        self.label_encoder: Optional[encoders.OneHotEncoder] = None

    def fit(self, X, y, **kwargs):
        # unite the input into pandas types
        X, y = utils.convert_inputs(X, y)
        y = pd.DataFrame(y.rename('target'))

        # apply one-hot-encoder on the label
        self.label_encoder = encoders.OneHotEncoder(handle_missing='error',
                                                    handle_unknown='error',
                                                    cols=['target'],
                                                    drop_invariant=True,
                                                    use_cat_names=True)
        labels = self.label_encoder.fit_transform(y)
        labels.columns = [column[7:] for column in labels.columns]
        labels = labels.iloc[:, 1:]  # drop one label

        # train the feature encoders, it is important to reset feature encoders first
        self.feature_encoders = {}
        for class_name, label in labels.items():
            self.feature_encoders[class_name] = copy.deepcopy(self.feature_encoder).fit(X, label)

    def transform(self, X, y=None):
        # unite the input into pandas types
        X = utils.convert_input(X)

        # initialization
        encoded = None
        feature_encoder = None
        all_new_features = pd.DataFrame()

        # transform the features
        if y is not None:
            y = self.label_encoder.transform(pd.DataFrame({"target": y}))
        for class_name, feature_encoder in self.feature_encoders.items():
            if y is not None:
                y_transform = y[f"target_{class_name}"]
            else:
                y_transform = None
            encoded = feature_encoder.transform(X, y_transform)

            # decorate the encoded features with the label class suffix
            new_features = encoded[feature_encoder.cols]
            new_features.columns = [str(column) + '_' + class_name for column in new_features.columns]

            all_new_features = pd.concat((all_new_features, new_features), axis=1)

        # add features that were not encoded
        result = pd.concat((encoded[encoded.columns[~encoded.columns.isin(feature_encoder.cols)]],
                            all_new_features), axis=1)

        return result

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y)


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
    >>> import pandas as pd
    >>> from category_encoders import *
    >>> from category_encoders.wrapper import NestedCVWrapper
    >>> from sklearn.datasets import fetch_openml
    >>> from sklearn.model_selection import GroupKFold, train_test_split
    >>> bunch = fetch_openml(name="house_prices", as_frame=True)
    >>> display_cols = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
    >>> y = bunch.target > 200000
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> X_train, X_test, y_train, _ = train_test_split(X, y, random_state=42)
    >>> X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)
    >>> # Define the nested CV encoder for a supervised encoder
    >>> enc_nested = NestedCVWrapper(TargetEncoder(cols=['CentralAir', 'Heating']), random_state=42)
    >>> # Encode the X data for train, valid & test
    >>> X_train_enc, X_valid_enc, X_test_enc = enc_nested.fit_transform(X_train, y_train, X_test=(X_valid, X_test))
    >>> print(X_train_enc.info())
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 821 entries, 1390 to 896
    Data columns (total 7 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Id           821 non-null    float64
     1   MSSubClass   821 non-null    float64
     2   MSZoning     821 non-null    object 
     3   LotFrontage  672 non-null    float64
     4   YearBuilt    821 non-null    float64
     5   Heating      821 non-null    float64
     6   CentralAir   821 non-null    float64
    dtypes: float64(6), object(1)
    memory usage: 51.3+ KB
    None
    """

    def __init__(self, feature_encoder, cv=5, shuffle=True, random_state=None):
        self.feature_encoder = feature_encoder
        self.__name__ = feature_encoder.__class__.__name__
        self.shuffle = shuffle
        self.random_state = random_state

        if isinstance(cv, int):
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
        out_of_fold = pd.DataFrame()

        for trn_idx, oof_idx in self.cv.split(X, y, groups):
            feature_encoder = copy.deepcopy(self.feature_encoder)
            feature_encoder.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            out_of_fold = pd.concat([out_of_fold, feature_encoder.transform(X.iloc[oof_idx])])

        # Train the encoder on all the training data for testing data
        self.feature_encoder = copy.deepcopy(self.feature_encoder)
        self.feature_encoder.fit(X, y)

        if X_test is None:
            return out_of_fold
        else:
            if isinstance(X_test, tuple):
                encoded_data = (out_of_fold, )
                for dataset in X_test:
                    encoded_data = encoded_data + (self.feature_encoder.transform(dataset), )
                return encoded_data
            else:
                return out_of_fold, self.feature_encoder.transform(X_test)
