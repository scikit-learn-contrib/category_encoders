v2.1.0
======

* Added experimental support for multithreading in hashing encoder
* Support for pandas >=0.24
* Removed support for missing values represented by None due to changes in Pandas 0.24. Use numpy.NaN
* Changed the default setting of Helmert encoder for handle_missing and handle_unknown
* Fixed wrong calculation in m-estimate encoder
* Fixed missing value handling in CatBoost encoder

v2.0.0
======

 * Added James-Stein, CatBoost and m-estimate encoders
 * Added get_feature_names method
 * Refactored treatment of missing and new values 
 * Speed up the encoders with vectorization
 * Improved compatibility with Pandas Series and Numpy Arrays

v1.3.0
======

 * Added Weight of Evidence encoder

v1.2.8
======

 * Critical bugfix in hashing encoder

v1.2.7
======

 * Bugfixes related to missing value imputation
 * Category names optionally added to encoded column names for some encoders
 * Documentation updates
 * Stats models pinned to avoid errors
 * Performance enhancements

v1.2.6
======

 * Release for zenodo DOI
 * Inverse transform implemented for some encoders

v1.2.5
======

 * Onehot transform returns same columns always
 * Missing value and unknown handling now configurable in all relevant encoders
 
v1.2.4
======

 * Added more sophisticated missing value or unknown category handling to ordinal
 * Passing through missing value config from onehot into ordinal
 * Onehot will return an extra column when unknown categories are passed in if impute is used.
 * Added BaseNEncoder to allow for more flexible alternatives to ordinal, onehot and binary.
 
v1.2.3
======

 * Full support for numpy arrays as input, not just dataframes.
 
v1.2.2
======

 * All encoders handle missing values and are tested for their handling
 * Created a onehot encoder that follows the same conventions as the rest of the library instead of using sklearns.
 * Did some basic benchmarking for data compression and memory usage, made some performance improvements
 * Changed all docstrings to numpy style and added more documentation
 * Moved all logic methods into staticmethods of the transformer classes themselves.
 * Added more detailed checks for type and shape of input data in fit and transform
 * Support input as list of lists, alongside numpy arrays and pandas dataframes.
 
v1.2.1
======

 * Better handling for missing values in hashing encoder
 
v1.2.0
======

 * Testing enhancements
 * Hash type in hashing encoder now defaults to md5 using hashlib, but can be set to any valid hashlib hash

v1.1.2
======

 * Added optional parameter to return a numpy array rather than a dataframe from all transformers.
 
v1.1.1
======

 * Immediately return if cols is empty.
 

v1.1.0
======

 * Optionally pass drop_invariant to any encoder to consistently drop columns with 0 variance from the output (based on training set data in fit())
 * If None is passed as the cols param, every string column will be encoded (pandas type = object).
 
v1.0.5
======

 * Changed setup.py to not explicitly force reinstalls of other packages
 
v1.0.4
======

 * Bugfixes
 
v1.0.0
======

 * First real usable release, includes sklearn compatible encoders.
 
v0.0.1
======

 * Basic library of encoders, no automated testing.