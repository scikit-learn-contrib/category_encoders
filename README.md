Categorical Encoding Methods
============================

[![Travis Status](https://travis-ci.org/scikit-learn-contrib/categorical-encoding.svg?branch=master)](https://travis-ci.org/scikit-learn-contrib/categorical-encoding)
[![Coveralls Status](https://coveralls.io/repos/scikit-learn-contrib/categorical-encoding/badge.svg?branch=master&service=github)](https://coveralls.io/r/scikit-learn-contrib/categorical-encoding)
[![CircleCI Status](https://circleci.com/gh/scikit-learn-contrib/categorical-encoding.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/scikit-learn-contrib/categorical-encoding/tree/master)
[![DOI](https://zenodo.org/badge/47077067.svg)](https://zenodo.org/badge/latestdoi/47077067)

A set of scikit-learn-style transformers for encoding categorical 
variables into numeric by means of different techniques.

Important Links
---------------

Documentation: [http://contrib.scikit-learn.org/categorical-encoding/](http://contrib.scikit-learn.org/categorical-encoding/)

Encoding Methods
----------------
__Unsupervised:__
 * Backward Difference Contrast [2][3]
 * BaseN [6]
 * Binary [5]
 * Count [10]
 * Hashing [1]
 * Helmert Contrast [2][3]
 * Ordinal [2][3]
 * One-Hot [2][3]
 * Polynomial Contrast [2][3]
 * Sum Contrast [2][3]

__Supervised:__
 * CatBoost [11]
 * James-Stein Estimator [9]
 * LeaveOneOut [4]
 * M-estimator [7]
 * Target Encoding [7]
 * Weight of Evidence [8]

Installation
------------

The package requires: `numpy`, `statsmodels`, and `scipy`.

To install the package, execute:

```shell
$ python setup.py install
```

or 

```shell
pip install category_encoders
```

or

```shell
conda install -c conda-forge category_encoders
```

To install the development version, you may use:

```shell
pip install --upgrade git+https://github.com/scikit-learn-contrib/categorical-encoding
```

Usage
-----

All of the encoders are fully compatible sklearn transformers, so they can be used in pipelines or in your existing scripts. Supported input formats include numpy arrays and pandas dataframes. If the cols parameter isn't passed, all columns with object or pandas categorical data type will be encoded. Please see the docs for transformer-specific configuration options.

Examples
--------
There are two types of encoders: unsupervised and supervised. An unsupervised example:
```python
from category_encoders import *
import pandas as pd
from sklearn.datasets import load_boston

# prepare some data
bunch = load_boston()
y = bunch.target
X = pd.DataFrame(bunch.data, columns=bunch.feature_names)

# use binary encoding to encode two categorical features
enc = BinaryEncoder(cols=['CHAS', 'RAD']).fit(X)

# transform the dataset
numeric_dataset = enc.transform(X)
```

And a supervised example:
```python
from category_encoders import *
import pandas as pd
from sklearn.datasets import load_boston

# prepare some data
bunch = load_boston()
y_train = bunch.target[0:250]
y_test = bunch.target[250:506]
X_train = pd.DataFrame(bunch.data[0:250], columns=bunch.feature_names)
X_test = pd.DataFrame(bunch.data[250:506], columns=bunch.feature_names)

# use target encoding to encode two categorical features
enc = TargetEncoder(cols=['CHAS', 'RAD'])

# transform the datasets
training_numeric_dataset = enc.fit_transform(X_train, y_train)
testing_numeric_dataset = enc.transform(X_test)
```

For the transformation of the _training_ data with the supervised methods, you should use `fit_transform()` method instead of `fit().transform()`, because these two methods _do not_ have to generate the same result. The difference can be observed with LeaveOneOut encoder, which performs a nested cross-validation for the _training_ data in `fit_transform()` method (to decrease over-fitting of the downstream model) but uses all the training data for scoring with `transform()` method (to get as accurate estimates as possible).

Additional examples and benchmarks can be found in the `examples` directory.

Contributing
------------

Category encoders is under active development, if you'd like to be involved, we'd love to have you. Check out the CONTRIBUTING.md file
or open an issue on the github project to get started.

References
----------

 1. Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola; Josh Attenberg (2009). Feature Hashing for Large Scale Multitask Learning. Proc. ICML.
 2. Contrast Coding Systems for categorical variables.  UCLA: Statistical Consulting Group. From https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/.
 3. Gregory Carey (2003). Coding Categorical Variables. From http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf
 4. Strategies to encode categorical variables with many categories. From https://www.kaggle.com/c/caterpillar-tube-pricing/discussion/15748#143154.
 5. Beyond One-Hot: an exploration of categorical variables. From http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/
 6. BaseN Encoding and Grid Search in categorical variables. From http://www.willmcginnis.com/2016/12/18/basen-encoding-grid-search-category_encoders/
 7. Daniele Miccii-Barreca (2001). A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems. SIGKDD Explor. Newsl. 3, 1. From http://dx.doi.org/10.1145/507533.507538
 8. Weight of Evidence (WOE) and Information Value Explained. From https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
 9. Empirical Bayes for multiple sample sizes. From http://chris-said.io/2017/05/03/empirical-bayes-for-multiple-sample-sizes/
 10. Simple Count or Frequency Encoding. From https://www.datacamp.com/community/tutorials/encoding-methodologies
 11. Transforming categorical features to numerical features. From https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/
 
