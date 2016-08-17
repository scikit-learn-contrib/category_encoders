.. Category Encoders documentation master file, created by
   sphinx-quickstart on Sat Jan 16 13:08:19 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Category Encoders
=================

A set of scikit-learn-style transformers for encoding categorical variables into numeric with different
techniques.  Currently implemented are:

 * Ordinal
 * One-Hot
 * Binary
 * Helmert Contrast
 * Sum Contrast
 * Polynomial Contrast
 * Backward Difference Contrast
 * Hashing

The ordinal, one-hot, and hashing encoders have similar equivalents in the existing scikit-learn version, but the
transformers in this library all share a few useful properties:

 * First-class support for pandas dataframes as an input (and optionally as output)
 * Can explicitly configure which columns in the data are encoded by name or index, or infer non-numeric columns regardless of input type
 * Can drop any columns with very low variance based on training set optionally
 * Portability: train a transformer on data, pickle it, reuse it later and get the same thing out.
 * Full compatibility with sklearn pipelines, input an array-like dataset like any other transformer

Usage
-----

install as:

.. code-block:: python

    pip install category_encoders

or

.. code-block:: python

    conda install -c conda-forge category_encoders


To use:

.. code-block:: python

    import category_encoders as ce

    encoder = ce.BackwardDifferenceEncoder(cols=[...])
    encoder = ce.BinaryEncoder(cols=[...])
    encoder = ce.HashingEncoder(cols=[...])
    encoder = ce.HelmertEncoder(cols=[...])
    encoder = ce.OneHotEncoder(cols=[...])
    encoder = ce.OrdinalEncoder(cols=[...])
    encoder = ce.SumEncoder(cols=[...])
    encoder = ce.PolynomialEncoder(cols=[...])

    encoder.fit(X, y)
    X_cleaned = encoder.transform(X_dirty)

All of these are fully compatible sklearn transformers, so they can be used in pipelines or in your existing scripts. If
the cols parameter isn't passed, every non-numeric column will be converted. See below for detailed documentation

Contents:

.. toctree::
   :maxdepth: 3

   backward_difference
   binary
   hashing
   helmert
   onehot
   ordinal
   polynomial
   sum


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

