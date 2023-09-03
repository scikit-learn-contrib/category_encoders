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
 * Full compatibility with sklearn pipelines, input an array-like dataset like any other transformer (\*)

(\*) For full compatibility with Pipelines and ColumnTransformers, and consistent behaviour of `get_feature_names_out`, it's recommended to upgrade `sklearn` to a version at least '1.2.0' and to set output as pandas:

.. code-block:: python

    import sklearn
    sklearn.set_config(transform_output="pandas")



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
    encoder = ce.CountEncoder(cols=[...])
    encoder = ce.GLMMEncoder(cols=[...])
    encoder = ce.GrayEncoder(cols=[...])
    encoder = ce.HashingEncoder(cols=[...])
    encoder = ce.HelmertEncoder(cols=[...])
    encoder = ce.JamesSteinEncoder(cols=[...])
    encoder = ce.LeaveOneOutEncoder(cols=[...])
    encoder = ce.MEstimateEncoder(cols=[...])
    encoder = ce.OneHotEncoder(cols=[...])
    encoder = ce.OrdinalEncoder(cols=[...])
    encoder = ce.PolynomialEncoder(cols=[...])
    encoder = ce.QuantileEncoder(cols=[...])
    encoder = ce.RankHotEncoder(cols=[...])
    encoder = ce.SumEncoder(cols=[...])
    encoder = ce.TargetEncoder(cols=[...])
    encoder = ce.WOEEncoder(cols=[...])

    encoder.fit(X, y)
    X_cleaned = encoder.transform(X_dirty)

All of these are fully compatible sklearn transformers, so they can be used in pipelines or in your existing scripts. If
the cols parameter isn't passed, every non-numeric column will be converted. See below for detailed documentation

Known issues:
----

`CategoryEncoders` internally works with `pandas DataFrames` as apposed to `sklearn` which works with `numpy arrays`. This can cause problems in `sklearn` versions prior to 1.2.0. In order to ensure full compatibility with `sklearn` set `sklearn` to also output `DataFrames`. This can be done by

.. code-block:: python

   sklearn.set_config(transform_output="pandas")

for a whole project or just for a single pipeline using

.. code-block:: python

   Pipeline(
       steps=[
           ("preprocessor", SomePreprocessor().set_output("pandas"),
           ("encoder", SomeEncoder()),
       ]
   )

If you experience another bug, feel free to report it on [github](https://github.com/scikit-learn-contrib/category_encoders/issues)

Contents:
----

.. toctree::
   :maxdepth: 3

   backward_difference
   basen
   binary
   catboost
   count
   glmm
   gray
   hashing
   helmert
   jamesstein
   leaveoneout
   mestimate
   onehot
   ordinal
   polynomial
   quantile
   rankhot
   sum
   summary
   targetencoder
   woe
   wrapper


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

