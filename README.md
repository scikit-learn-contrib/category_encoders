Categorical Encoding Methods
============================

[![Travis Status](https://travis-ci.org/scikit-learn-contrib/categorical-encoding.svg?branch=master)](https://travis-ci.org/scikit-learn-contrib/categorical-encoding)
[![Coveralls Status](https://coveralls.io/repos/scikit-learn-contrib/categorical-encoding/badge.svg?branch=master&service=github)](https://coveralls.io/r/scikit-learn-contrib/categorical-encoding)
[![CircleCI Status](https://circleci.com/gh/scikit-learn-contrib/categorical-encoding.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/scikit-learn-contrib/categorical-encoding/tree/master)

A set of scikit-learn-style transformers for encoding categorical 
variables into numeric by means of different techniques.

Important Links
---------------

Documentation: [http://contrib.scikit-learn.org/categorical-encoding/](http://contrib.scikit-learn.org/categorical-encoding/)

Encoding Methods
----------------

 * Ordinal [2][3]
 * One-Hot [2][3]
 * Binary
 * Helmert Contrast [2][3]
 * Sum Contrast [2][3]
 * Polynomial Contrast [2][3]
 * Backward Difference Contrast [2][3]
 * Hashing [1]
 * BaseN

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
    encoder = ce.BinaryEncoder(cols=[...])
    encoder = ce.HashingEncoder(cols=[...])
    encoder = ce.HelmertEncoder(cols=[...])
    encoder = ce.OneHotEncoder(cols=[...])
    encoder = ce.OrdinalEncoder(cols=[...])
    encoder = ce.SumEncoder(cols=[...])
    encoder = ce.PolynomialEncoder(cols=[...])
    encoder = ce.BaseNEncoder(cols=[...])

All of these are fully compatible sklearn transformers, so they can be used in pipelines or in your existing scripts. If 
the cols parameter isn't passed, every non-numeric column will be encoded. Please see the 
docs for transformer-specific configuration options.

Examples
--------

In the examples directory, there is an example script used to benchmark
different encoding techniques on various datasets.

The datasets used in the examples are car, mushroom, and splice datasets 
from the UCI dataset repository, found here:

[datasets](https://archive.ics.uci.edu/ml/datasets)

License
-------

BSD 3-Clause

References:
-----------

 1. Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola; Josh Attenberg (2009). Feature Hashing for Large Scale Multitask Learning. Proc. ICML.
 2. Contrast Coding Systems for categorical variables.  UCLA: Statistical Consulting Group. from http://www.ats.ucla.edu/stat/r/library/contrast_coding.
 3. Gregory Carey (2003). Coding Categorical Variables, from http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf
 
