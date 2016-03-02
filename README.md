Categorical Encoding Methods
============================

A set of example problems examining different encoding methods for categorical variables for the purpose of 
classification. Optionally, install the library of encoders as a package and use them in your projects directly.  They 
are all available as methods or as scikit-learn compatible transformers. 

Docs [here](http://wdm0006.github.io/categorical_encoding/)

Encoding Methods
----------------

 * Ordinal
 * One-Hot
 * Binary
 * Helmert Contrast
 * Sum Contrast
 * Polynomial Contrast
 * Backward Difference Contrast
 * Simple Hashing

Usage
-----

Either run the examples in encoding_examples.py, or install as:

    pip install category_encoders
    
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
the cols parameter isn't passed, every column will be encoded, so be careful with that.

Datasets
--------

The datasets used in the examples are car, mushroom, and splice datasets from the UCI dataset repository, found here:

[datasets](https://archive.ics.uci.edu/ml/datasets)

License
-------

BSD