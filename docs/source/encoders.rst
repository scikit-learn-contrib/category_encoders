Encoders
========

A listing of the encoders included in this library, specifically focused on the sklearn compatible transformers.  For
each one of these, an equivalent standalone method is also available.

Ordinal
-------

.. autoclass:: category_encoders.ordinal.OrdinalEncoder
    :members:

One-Hot
-------

Scikit-learn already includes a one-hot encoder implementation, and we simply expose that as an option here. See their
docs for detailed information.

Binary
------

.. autoclass:: category_encoders.binary.BinaryEncoder
    :members:


Hashing
-------

.. autoclass:: category_encoders.hashing.HashingEncoder
    :members:


Backward Difference Coding
--------------------------

.. autoclass:: category_encoders.backward_difference.BackwardDifferenceEncoder
    :members:


Helmert Coding
--------------

.. autoclass:: category_encoders.helmert.HelmertEncoder
    :members:

Polynomial Coding
-----------------

.. autoclass:: category_encoders.polynomial.PolynomialEncoder
    :members:


Sum Coding
----------

.. autoclass:: category_encoders.sum_coding.SumEncoder
    :members:

