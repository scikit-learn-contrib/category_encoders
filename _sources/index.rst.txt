.. Category Encoders documentation master file, created by
   sphinx-quickstart on Sat Jan 16 13:08:19 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Category Encoders
=================

A set of scikit-learn-style transformers for encoding categorical variables into numeric with different
techniques. While ordinal, one-hot, and hashing encoders have similar equivalents in the existing scikit-learn version, the
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
    encoder = ce.BaseNEncoder(cols=[...])
    encoder = ce.BinaryEncoder(cols=[...])
    encoder = ce.CatBoostEncoder(cols=[...])
    encoder = ce.HashingEncoder(cols=[...])
    encoder = ce.HelmertEncoder(cols=[...])
    encoder = ce.JamesSteinEncoder(cols=[...])
    encoder = ce.LeaveOneOutEncoder(cols=[...])
    encoder = ce.MEstimateEncoder(cols=[...])
    encoder = ce.OneHotEncoder(cols=[...])
    encoder = ce.OrdinalEncoder(cols=[...])
    encoder = ce.SumEncoder(cols=[...])
    encoder = ce.PolynomialEncoder(cols=[...])
    encoder = ce.TargetEncoder(cols=[...])
    encoder = ce.WOEEncoder(cols=[...])

    encoder.fit(X, y)
    X_cleaned = encoder.transform(X_dirty)

All of these are fully compatible sklearn transformers, so they can be used in pipelines or in your existing scripts. If
the cols parameter isn't passed, every non-numeric column will be converted. See below for detailed documentation

Contents:

.. toctree::
   :maxdepth: 3

   backward_difference
   basen
   binary
   catboost
   hashing
   helmert
   jamesstein
   leaveoneout
   mestimate
   onehot
   ordinal
   polynomial
   sum
   targetencoder
   woe


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

