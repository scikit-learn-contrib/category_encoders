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

 * Backward Difference Contrast [2][3]
 * BaseN [6]
 * Binary [5]
 * Hashing [1]
 * Helmert Contrast [2][3]
 * LeaveOneOut [4]
 * Ordinal [2][3]
 * One-Hot [2][3]
 * Polynomial Contrast [2][3]
 * Sum Contrast [2][3]
 * Target Encoding [7]
 * Weight of Evidence [8]

Usage
-----

The package by itself comes with a single module and an estimator. Before
installing the module you will need `numpy`, `statsmodels`, and `scipy`.

To install the module execute:

```shell
$ python setup.py install
```

or 

```
pip install category_encoders
```

or

```
conda install -c conda-forge category_encoders
```
    
To use:

    import category_encoders as ce
    
    encoder = ce.BackwardDifferenceEncoder(cols=[...])
    encoder = ce.BaseNEncoder(cols=[...])
    encoder = ce.BinaryEncoder(cols=[...])
    encoder = ce.HashingEncoder(cols=[...])
    encoder = ce.HelmertEncoder(cols=[...])
    encoder = ce.LeaveOneOutEncoder(cols=[...])
    encoder = ce.OneHotEncoder(cols=[...])
    encoder = ce.OrdinalEncoder(cols=[...])
    encoder = ce.PolynomialEncoder(cols=[...])
    encoder = ce.SumEncoder(cols=[...])
    encoder = ce.TargetEncoder(cols=[...])
    encoder = ce.WOEEncoder(cols=[...])

All of these are fully compatible sklearn transformers, so they can be used in pipelines or in your existing scripts. If 
the cols parameter isn't passed, all pandas columns with the object and categorical data type will be encoded. Please see the 
docs for transformer-specific configuration options.

Examples
--------

    from category_encoders import *
    import pandas as pd
    from sklearn.datasets import load_boston

    # prepare some data
    bunch = load_boston()
    y = bunch.target
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)

    # use binary encoding to encode two categorical features
    enc = BinaryEncoder(cols=['CHAS', 'RAD']).fit(X, y)

    # transform the dataset
    numeric_dataset = enc.transform(X)

In the examples directory, there is an example script used to benchmark
different encoding techniques on various datasets.

The datasets used in the examples are car, mushroom, and splice datasets 
from the UCI dataset repository, found here:

[datasets](https://archive.ics.uci.edu/ml/datasets)

Contributing
------------

Category encoders is under active development, if you'd like to be involved, we'd love to have you. Check out the CONTRIBUTING.md file
or open an issue on the github project to get started.

License
-------

BSD 3-Clause

References:
-----------

 1. Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola; Josh Attenberg (2009). Feature Hashing for Large Scale Multitask Learning. Proc. ICML.
 2. Contrast Coding Systems for categorical variables.  UCLA: Statistical Consulting Group. from https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/.
 3. Gregory Carey (2003). Coding Categorical Variables. from http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf
 4. Strategies to encode categorical variables with many categories. from https://www.kaggle.com/c/caterpillar-tube-pricing/discussion/15748#143154.
 5. Beyond One-Hot: an exploration of categorical variables. from http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/
 6. BaseN Encoding and Grid Search in categorical variables. from http://www.willmcginnis.com/2016/12/18/basen-encoding-grid-search-category_encoders/
 7. A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems. from https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
 8. Weight of Evidence (WOE) and Information Value Explained. from https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
