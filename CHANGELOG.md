unreleased
==========

* added: ignore option for one-hot-encoding
* fixed: external dependency in unit test
* fixed: gaps in ordinal encoding if nan values are present
* fixed: sklearn complicance: add `feature_names_in_` attribute
* fixed: add RankHotEncoder in documentation
* fixed: return correct mapping in one hot encoder `category_mapping` property (issue #256)

v2.6.0
======
* added gray encoder
* added thermometer / rank-hot encoder
* introduce compatibility with sklearn 1.2
  * compatibility with `feature_names_out_`
  * remove boston housing dataset
  * drop support for dataframes with non-homogenous data types in column names (i.e. having both string and integer column names)
* improve performance of hashing encoder
* improve catboost documentation
* fix inverse transform in baseN with special character column names (issue 392)
* fix inverse transform of ordinal encoder with custom mapping (issue 202)
* fix re-fittable polynomial wrapper (issue 313)
* fix numerical stability for target encoding (issue 377)
* change default parameters of target encoding (issue 327)
* drop support for sklearn 0.x
 
v2.5.1.post0
============
* fix pypi sdist

v2.5.1
======
* Added base class for contrast coding schemes in order to make them more maintainable
* Added hierarchical column feature in target encoder
* Fixed maximum recursion depth bug in hashing encoder

v2.5.0
======

* Introduce base class for encoders
* Introduce tagging system on encoders and use it to parametrize tests
* Drop support for python 3.5 and python 3.6
* Require pandas >=1.0
* Introduce f-strings
* Make BinaryEncoder a BaseNEncoder for base=2
* FutureWarning for TargetEncoder's default parameters
* Made all encoders re-fittable on different datasets (c.f. issue 122)
* Introduced tox.ini file for easier version testing
 
v2.4.1
======

* Fixed a bug with categorical data type in LeaveOneOut encoder
* Do not install examples as a package on its own
 
v2.4.0
======

* improved documentation
* fix bug in CatBoost encoder
* fix future warnings with pandas
* added tests for python 3.9 and 3.10 in pipeline
* fix treating np.NaN and python None equal
* only build docs on release
* unified conversion of inputs pandas objects that are used internally including some bugfixes.
* added quantile encoder and summary encoder
 
v2.3.0
======

 * many bugfixes
 * added count encoder
 
v2.2.2
======
* Added generalized linear mixed model encoder
* Added cross-validation wrapper
* Added multi-class wrapper
* Support for pandas >= 1.0.1
* Moved CI to github actions

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
