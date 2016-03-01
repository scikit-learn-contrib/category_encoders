.. Category Encoders documentation master file, created by
   sphinx-quickstart on Sat Jan 16 13:08:19 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Category Encoders
=================

A set of example problems examining different encoding methods for categorical variables for the purpose of
classification. Optionally, install the library of encoders as a package and use them in your projects directly.  They
are all available as methods or as scikit-learn compatible transformers.

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

Contents:

.. toctree::
   :maxdepth: 2

   encoders


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

