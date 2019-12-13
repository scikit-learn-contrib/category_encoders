"""The hashing module contains all methods and classes related to the hashing trick."""

import sys
import hashlib
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders.utils as util
import multiprocessing
import pandas as pd
import math
import platform

__author__ = 'willmcginnis', 'LiuShulun'


class HashingEncoder(BaseEstimator, TransformerMixin):

    """ A multivariate hashing implementation with configurable dimensionality/precision.

    The advantage of this encoder is that it does not maintain a dictionary of observed categories.
    Consequently, the encoder does not grow in size and accepts new values during data scoring
    by design.

    It's important to read about how max_process & max_sample work
    before setting them manually, inappropriate setting slows down encoding.

    Default value of 'max_process' is 1 on Windows because multiprocessing might cause issues, see in :
    https://github.com/scikit-learn-contrib/categorical-encoding/issues/215
    https://docs.python.org/2/library/multiprocessing.html?highlight=process#windows

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    hash_method: str
        which hashing method to use. Any method from hashlib works.
    max_process: int
        how many processes to use in transform(). Limited in range(1, 64).
        By default, it uses half of the logical CPUs.
        For example, 4C4T makes max_process=2, 4C8T makes max_process=4.
        Set it larger if you have a strong CPU.
        It is not recommended to set it larger than is the count of the
        logical CPUs as it will actually slow down the encoding.
    max_sample: int
        how many samples to encode by each process at a time.
        This setting is useful on low memory machines.
        By default, max_sample=(all samples num)/(max_process).
        For example, 4C8T CPU with 100,000 samples makes max_sample=25,000,
        6C12T CPU with 100,000 samples makes max_sample=16,666.
        It is not recommended to set it larger than the default value.
    n_components: int
        how many bits to use to represent the feature. By default we use 8 bits.
        For high-cardinality features, consider using up-to 32 bits.

    Example
    -------
    >>> from category_encoders.hashing import HashingEncoder
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> y = bunch.target
    >>> he = HashingEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> data = he.transform(X)
    >>> print(data.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 19 columns):
    col_0      506 non-null int64
    col_1      506 non-null int64
    col_2      506 non-null int64
    col_3      506 non-null int64
    col_4      506 non-null int64
    col_5      506 non-null int64
    col_6      506 non-null int64
    col_7      506 non-null int64
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(11), int64(8)
    memory usage: 75.2 KB
    None

    References
    ----------
    .. [1] Feature Hashing for Large Scale Multitask Learning, from
    https://alex.smola.org/papers/2009/Weinbergeretal09.pdf
    .. [2] Don't be tricked by the Hashing Trick, from
    https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087

    """

    def __init__(self, max_process=0, max_sample=0, verbose=0, n_components=8, cols=None, drop_invariant=False, return_df=True, hash_method='md5'):

        if max_process not in range(1, 128):
            if platform.system == 'Windows':
                max_process = 1
            else:
                self.max_process = int(math.ceil(multiprocessing.cpu_count() / 2))
                if self.max_process < 1:
                    self.max_process = 1
                elif self.max_process > 128:
                    self.max_process = 128
        else:
            self.max_process = max_process
        self.max_sample = int(max_sample)
        self.auto_sample = max_sample <= 0
        self.data_lines = 0
        self.X = None

        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.n_components = n_components
        self.cols = cols
        self.hash_method = hash_method
        self._dim = None
        self.feature_names = None

    def fit(self, X, y=None, **kwargs):
        """Fit encoder according to X and y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # first check the type
        X = util.convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                          "Not found in generated cols.\n{}".format(e))

        return self

    @staticmethod
    def require_data(self, data_lock, new_start, done_index, hashing_parts, cols, process_index):
        if data_lock.acquire():
            if new_start.value:
                end_index = 0
                new_start.value = False
            else:
                end_index = done_index.value

            if all([self.data_lines > 0, end_index < self.data_lines]):
                start_index = end_index
                if (self.data_lines - end_index) <= self.max_sample:
                    end_index = self.data_lines
                else:
                    end_index += self.max_sample
                done_index.value = end_index
                data_lock.release()

                data_part = self.X.iloc[start_index: end_index]
                # Always get df and check it after merge all data parts
                data_part = self.hashing_trick(X_in=data_part, hashing_method=self.hash_method, N=self.n_components, cols=self.cols)
                if self.drop_invariant:
                    for col in self.drop_cols:
                        data_part.drop(col, 1, inplace=True)
                part_index = int(math.ceil(end_index / self.max_sample))
                hashing_parts.put({part_index: data_part})
                if self.verbose == 5:
                    print("Process - " + str(process_index),
                          "done hashing data : " + str(start_index) + "~" + str(end_index))
                if end_index < self.data_lines:
                    self.require_data(self, data_lock, new_start, done_index, hashing_parts, cols=cols, process_index=process_index)
            else:
                data_lock.release()
        else:
            data_lock.release()

    def transform(self, X, override_return_df=False):
        """
        Call _transform() if you want to use single CPU with all samples
        """
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        self.X = util.convert_input(X)
        self.data_lines = len(self.X)

        # then make sure that it is the right size
        if self.X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (self.X.shape[1], self._dim, ))

        if not list(self.cols):
            return self.X

        data_lock = multiprocessing.Manager().Lock()
        new_start = multiprocessing.Manager().Value('d', True)
        done_index = multiprocessing.Manager().Value('d', int(0))
        hashing_parts = multiprocessing.Manager().Queue()

        if self.auto_sample:
            self.max_sample = int(self.data_lines / self.max_process)
        if self.max_process == 1:
            self.require_data(self, data_lock, new_start, done_index, hashing_parts, cols=self.cols, process_index=1)
        else:
            n_process = []
            for thread_index in range(self.max_process):
                process = multiprocessing.Process(target=self.require_data,
                                                  args=(self, data_lock, new_start, done_index, hashing_parts, self.cols, thread_index + 1))
                process.daemon = True
                n_process.append(process)
            for process in n_process:
                process.start()
            for process in n_process:
                process.join()
        data = self.X
        if self.max_sample == 0 or self.max_sample == self.data_lines:
            if hashing_parts:
                data = list(hashing_parts.get().values())[0]
        else:
            list_data = {}
            while not hashing_parts.empty():
                list_data.update(hashing_parts.get())
            sort_data = []
            for part_index in sorted(list_data):
                sort_data.append(list_data[part_index])
            if sort_data:
                data = pd.concat(sort_data, ignore_index=True)
        # Check if is_return_df
        if self.return_df or override_return_df:
            return data
        else:
            return data.values

    def _transform(self, X, override_return_df=False):
        """Perform the transformation to new categorical data.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]

        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim, ))

        if not list(self.cols):
            return X

        X = self.hashing_trick(X, hashing_method=self.hash_method, N=self.n_components, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    @staticmethod
    def hashing_trick(X_in, hashing_method='md5', N=2, cols=None, make_copy=False):
        """A basic hashing implementation with configurable dimensionality/precision

        Performs the hashing trick on a pandas dataframe, `X`, using the hashing method from hashlib
        identified by `hashing_method`.  The number of output dimensions (`N`), and columns to hash (`cols`) are
        also configurable.

        Parameters
        ----------

        X_in: pandas dataframe
            description text
        hashing_method: string, optional
            description text
        N: int, optional
            description text
        cols: list, optional
            description text
        make_copy: bool, optional
            description text

        Returns
        -------

        out : dataframe
            A hashing encoded dataframe.

        References
        ----------
        Cite the relevant literature, e.g. [1]_.  You may also cite these
        references in the notes section above.
        .. [1] Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola; Josh Attenberg (2009). Feature Hashing
        for Large Scale Multitask Learning. Proc. ICML.

        """

        try:
            if hashing_method not in hashlib.algorithms_available:
                raise ValueError('Hashing Method: %s Not Available. Please use one from: [%s]' % (
                    hashing_method,
                    ', '.join([str(x) for x in hashlib.algorithms_available])
                ))
        except Exception as e:
            try:
                _ = hashlib.new(hashing_method)
            except Exception as e:
                raise ValueError('Hashing Method: %s Not Found.')

        if make_copy:
            X = X_in.copy(deep=True)
        else:
            X = X_in

        if cols is None:
            cols = X.columns.values

        def hash_fn(x):
            tmp = [0 for _ in range(N)]
            for val in x.values:
                if val is not None:
                    hasher = hashlib.new(hashing_method)
                    if sys.version_info[0] == 2:
                        hasher.update(str(val))
                    else:
                        hasher.update(bytes(str(val), 'utf-8'))
                    tmp[int(hasher.hexdigest(), 16) % N] += 1
            return pd.Series(tmp, index=new_cols)

        new_cols = ['col_%d' % d for d in range(N)]

        X_cat = X.loc[:, cols]
        X_num = X.loc[:, [x for x in X.columns.values if x not in cols]]

        X_cat = X_cat.apply(hash_fn, axis=1)
        X_cat.columns = new_cols

        X = pd.concat([X_cat, X_num], axis=1)

        return X

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """

        if not isinstance(self.feature_names, list):
            raise ValueError('Must fit data first. Affected feature names are not known before.')
        else:
            return self.feature_names
