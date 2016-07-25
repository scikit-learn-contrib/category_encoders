.. Category Encoders documentation master file, created by
   sphinx-quickstart on Sat Jan 16 13:08:19 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Category Encoders
=================

A set of scikit-learn-style transformers for encoding categorical variables into numeric by means of different
techniques.

Usage
-----

Either run the examples in encoding_examples.py, or install as:

.. code-block:: python

    pip install category_encoders

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

All of these are fully compatible sklearn transformers, so they can be used in pipelines or in your existing scripts. If
the cols parameter isn't passed, every column will be encoded, so be careful with that. See below for detailed documentation

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

