Categorical Encoding Methods
============================

[![Travis Status](https://travis-ci.org/wdm0006/categorical_encoding.svg?branch=master)](https://travis-ci.org/wdm0006/categorical_encoding)
[![Coveralls Status](https://coveralls.io/repos/wdm0006/categorical_encoding/badge.svg?branch=master&service=github)](https://coveralls.io/r/wdm0006/categorical_encoding)
[![CircleCI Status](https://circleci.com/gh/wdm0006/categorical_encoding.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/wdm0006/categorical_encoding/tree/master)

A set of scikit-learn-style transformers for encoding categorical 
variables into numeric by means of different techniques.

Important Links
---------------

Documentation: [http://wdm0006.github.io/categorical_encoding/](http://wdm0006.github.io/categorical_encoding/)

Encoding Methods
----------------

 * Ordinal
 * One-Hot
 * Binary
 * Helmert Contrast
 * Sum Contrast
 * Polynomial Contrast
 * Backward Difference Contrast
 * Hashing

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

BSD