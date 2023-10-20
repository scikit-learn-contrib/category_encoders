"""The hashing module contains all methods and classes related to the hashing trick."""

import sys
import hashlib
import category_encoders.utils as util
import multiprocessing
import pandas as pd
import numpy as np
import math
import platform

__author__ = 'willmcginnis', 'LiuShulun'


class HashingEncoder(util.BaseEncoder, util.UnsupervisedTransformerMixin):

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
        how many bits to use to represent the feature. By default, we use 8 bits.
        For high-cardinality features, consider using up-to 32 bits.
    process_creation_method: string
        either "fork", "spawn" or "forkserver" (availability depends on your 
        platform). See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        for more details and tradeoffs. Defaults to "fork" on linux/macos as it
        is the fastest option and to "spawn" on windows as it is the only one
        available

    Example
    -------
    >>> from category_encoders.hashing import HashingEncoder
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name="house_prices", as_frame=True)
    >>> display_cols = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> y = bunch.target
    >>> he = HashingEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
    >>> numeric_dataset = he.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 13 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   col_0        1460 non-null   int64  
     1   col_1        1460 non-null   int64  
     2   col_2        1460 non-null   int64  
     3   col_3        1460 non-null   int64  
     4   col_4        1460 non-null   int64  
     5   col_5        1460 non-null   int64  
     6   col_6        1460 non-null   int64  
     7   col_7        1460 non-null   int64  
     8   Id           1460 non-null   float64
     9   MSSubClass   1460 non-null   float64
     10  MSZoning     1460 non-null   object 
     11  LotFrontage  1201 non-null   float64
     12  YearBuilt    1460 non-null   float64
    dtypes: float64(4), int64(8), object(1)
    memory usage: 148.4+ KB
    None

    References
    ----------
    .. [1] Feature Hashing for Large Scale Multitask Learning, from
    https://alex.smola.org/papers/2009/Weinbergeretal09.pdf
    .. [2] Don't be tricked by the Hashing Trick, from
    https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087

    """
    prefit_ordinal = False
    encoding_relation = util.EncodingRelation.ONE_TO_M
    default_int_np_array = np.array(np.zeros((2,2), dtype='int'))

    def __init__(self, max_process=0, max_sample=0, verbose=0, n_components=8, cols=None, drop_invariant=False,
                 return_df=True, hash_method='md5', process_creation_method='fork'):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown="does not apply", handle_missing="does not apply")

        if max_process not in range(1, 128):
            if platform.system() == 'Windows':
                self.max_process = 1
            else:
                self.max_process = int(math.ceil(multiprocessing.cpu_count() / 2))
                if self.max_process < 1:
                    self.max_process = 1
                elif self.max_process > 128:
                    self.max_process = 128
        else:
            self.max_process = max_process
        self.max_sample = int(max_sample)
        if platform.system() == 'Windows':
            self.process_creation_method = "spawn"
        else:
            self.process_creation_method = process_creation_method
        self.data_lines = 0
        self.X = None

        self.n_components = n_components
        self.hash_method = hash_method

    def _fit(self, X, y=None, **kwargs):
        pass

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
            raise ValueError(f'Unexpected input dimension {X.shape[1]}, expected {self._dim}')

        if not list(self.cols):
            return X

        X = self.hashing_trick(
            X, 
            hashing_method=self.hash_method,
            N=self.n_components, 
            cols=self.cols, 
        )
        
        return X

    @staticmethod
    def hash_chunk(hash_method, shared_memory_result, np_df, N, shared_memory_offset):
        # Calling getattr outside the loop saves some time in the loop
        hasher_constructor = getattr(hashlib, hash_method)
        # Same when the call to getattr is implicit
        int_from_bytes = int.from_bytes
        for i, row in enumerate(np_df):
            for val in row:
                if val is not None:
                    hasher = hasher_constructor()
                    hasher.update(bytes(str(val), 'utf-8'))
                    # Computes an integer index from the hasher digest. The endian is 
                    # "big" as the code use to read:
                    # column_index = int(hasher.hexdigest(), 16) % N
                    # which is implicitly considering the hexdigest to be big endian,
                    # even if the system is little endian.
                    # Building the index that way is about 30% faster than using the 
                    # hexdigest.
                    column_index = int_from_bytes(hasher.digest(), byteorder='big') % N
                    row_index = (shared_memory_offset + i)*N
                    shared_memory_index = row_index + column_index
                    shared_memory_result[shared_memory_index] += 1

    def hashing_trick_with_np_parallel(self, df, N):
        np_df = df.to_numpy()
        shared_memory_result = multiprocessing.RawArray(HashingEncoder.default_int_np_array.dtype.char, len(df)*N)

        process_list = []
        chunk_size = int(len(np_df)/self.max_process)
        ctx = multiprocessing.get_context(self.process_creation_method)
        for i in range(0, self.max_process-1):
            process = ctx.Process(target=self.hash_chunk,
                args=(self.hash_method, shared_memory_result, np_df[i*chunk_size:((i+1)*chunk_size)], N, i*chunk_size))
            process_list.append(process)

        # The last process processes all the rest of the dataframe, because the number of rows might not
        # be divisible by max_process. 
        process = ctx.Process(target=self.hash_chunk,
            args=(self.hash_method, shared_memory_result, np_df[(self.max_process-1)*chunk_size:], N, (self.max_process-1)*chunk_size))
        process_list.append(process)

        for process in process_list:
            process.start()
        for process in process_list:
            process.join()

        np_result = np.array(shared_memory_result, 'int')

        return pd.DataFrame(np_result.reshape(len(df), N), index=df.index)

    def hashing_trick_with_np_no_parallel(self, df, N):
        np_df = df.to_numpy()
        np_result = np.zeros((len(df)*N), dtype='int')

        HashingEncoder.hash_chunk(self.hash_method, np_result, np_df, N, 0)
        
        return pd.DataFrame(np_result.reshape(len(df), N), index=df.index)

    def hashing_trick(self, X_in, hashing_method='md5', N=2, cols=None, make_copy=False):
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
        if hashing_method not in hashlib.algorithms_available:
            raise ValueError(f"Hashing Method: {hashing_method} not Available. "
                             f"Please use one from: [{', '.join([str(x) for x in hashlib.algorithms_available])}]")

        if make_copy:
            X = X_in.copy(deep=True)
        else:
            X = X_in

        if cols is None:
            cols = X.columns

        new_cols = [f'col_{d}' for d in range(N)]

        X_cat = X.loc[:, cols]
        X_num = X.loc[:, [x for x in X.columns if x not in cols]]

        if self.max_process == 1:
            X_cat = self.hashing_trick_with_np_no_parallel(X_cat, N)
        else:
            X_cat = self.hashing_trick_with_np_parallel(X_cat, N)

        X_cat.columns = new_cols

        X = pd.concat([X_cat, X_num], axis=1)

        return X
