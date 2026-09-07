How category_encoders differs from scikit-learn
===============================================

The encoders in this library follow the scikit-learn estimator pattern: they are
configured in the constructor, ``fit`` learns the encoding from training data,
and ``transform`` applies it to new data. They work with
``sklearn.pipeline.Pipeline``, ``get_params``/``set_params``, pickling, and
model-inspection tools.

They are nevertheless not drop-in replacements for the scikit-learn transformers
they resemble. This page describes the conventions that differ, so that
pipelines, cross-validation, and custom wrappers behave as expected.

The target is required to fit
-----------------------------

Most encoders in this library are *supervised*: the encoding is learned from the
relationship between each categorical column and the target ``y``. For these,
``y`` is a required argument of ``fit`` — fitting without it raises a
``ValueError`` ("Supervised encoders need a target for the fitting"), and ``y``
must not contain missing values.

The supervised encoders are: ``CatBoostEncoder``, ``GLMMEncoder``,
``JamesSteinEncoder``, ``LeaveOneOutEncoder``, ``MEstimateEncoder``,
``QuantileEncoder``, ``SummaryEncoder``, ``TargetEncoder`` and
``WOEEncoder``. All other encoders are unsupervised: they accept ``fit(X)``,
ignore ``y`` if one is passed, and their ``transform`` takes no ``y`` at all.

A non-numeric ``y`` does not need to be pre-encoded: supervised encoders fit an
internal :class:`sklearn.preprocessing.LabelEncoder` (stored as
``lab_encoder_``) on string or boolean targets.

``transform`` may use the target
--------------------------------

In scikit-learn, ``transform`` never sees ``y``. Here, supervised encoders
*may* be called with one:

.. code-block:: python

    encoder.fit(X_train, y_train)
    X_train_encoded = encoder.transform(X_train, y_train)  # training data
    X_test_encoded = encoder.transform(X_test)             # test data

Passing ``y`` on training data lets the encoder apply its regularization (for
example leave-one-out or smoothing statistics) instead of the unregularized
mapping. Related to this, ``fit_transform`` is not merely a ``fit`` followed by
an untargeted ``transform``: it requires ``y`` (a ``TypeError`` is raised
without it) and uses the target for transforming as well.

Missing and unknown values default to ``'value'``
-------------------------------------------------

By default, missing values and categories unseen at fit time are not errors —
they are treated as countable categories and encoded like any other value
(``handle_missing='value'`` and ``handle_unknown='value'``).
scikit-learn's ``OneHotEncoder`` instead raises by default on unseen categories
at transform time (``handle_unknown='error'``).

Both ``handle_missing`` and ``handle_unknown`` accept the strings ``'error'``
(raise), ``'return_nan'`` (propagate NaN), and ``'value'`` (encode as a
category). The strategies are validated at fit time; an unrecognized string
raises a ``ValueError`` naming the supported values. Two families extend the
base set:

* the one-hot, base-N, and contrast families (``OneHotEncoder``,
  ``BaseNEncoder``, ``BinaryEncoder``, ``GrayEncoder``,
  ``BackwardDifferenceEncoder``, ``HelmertEncoder``, ``PolynomialEncoder``,
  ``SumEncoder``) additionally accept ``'indicator'``, which reserves an extra
  column for unknown or missing values;
* ``CountEncoder`` and ``HashingEncoder`` accept additional non-string values
  and do not validate these two parameters against the string set.

Note that scikit-learn's ``OneHotEncoder`` has no ``handle_missing`` parameter
at all: it always treats NaN as its own category during fitting.

Output type: a DataFrame by default
-----------------------------------

scikit-learn transformers return NumPy arrays unless configured otherwise. Here,
``transform`` returns a pandas DataFrame by default (``return_df=True``);
passing ``return_df=False`` at construction yields a NumPy array instead.
Output DataFrames carry the input index and the fitted output column names
(see ``get_feature_names_out`` below).

``set_output`` works as usual: encoders inherit it from
:class:`sklearn.base.TransformerMixin`, so
``encoder.set_output(transform="pandas")`` (or the global
``sklearn.set_config(transform_output="pandas")``) integrates them into
pipelines that mix DataFrame- and array-valued steps.

Column selection happens inside the estimator
---------------------------------------------

scikit-learn selects which columns a transformer applies to *outside* of the
estimator, typically via :class:`sklearn.compose.ColumnTransformer`. Here, the
selection is part of the estimator itself, via the ``cols`` parameter:

* ``cols=None`` (default): all columns with an object, category, or string
  dtype are encoded, numeric columns pass through untouched;
* ``cols='all'``: every column is encoded regardless of dtype;
* ``cols=['a', 'b']``: exactly the named columns are encoded.

``fit`` records the input width (``n_features_in_``), and ``transform``
raises a ``ValueError`` ("Unexpected input dimension ...") when called on data
with a different number of columns.

Feature names require a fit
---------------------------

``get_feature_names_out()`` returns the names of the encoded output columns,
but — unlike in scikit-learn, where names can be derived from
``input_features`` without fitting — here the encoder must be fitted first;
calling it before ``fit`` raises a ``NotFittedError``. The deprecated
``get_feature_names`` method forwards to it with a ``FutureWarning``.

The fitted attributes follow the scikit-learn naming convention:
``feature_names_in_``, ``n_features_in_`` and ``feature_names_out_``.

``X`` and ``y`` indexes are aligned
-----------------------------------

scikit-learn treats inputs as positional arrays. When ``X`` and ``y`` are both
pandas objects, this library aligns them by index: if the two indexes do not
match, ``fit`` (and ``transform``) raise a ``ValueError`` suggesting to use
NumPy arrays when the data is intentionally shuffled (for example inside
``sklearn.model_selection.permutation_test_score``). This catches
row-misalignment bugs that positional libraries cannot see — at the cost of
rejecting inputs scikit-learn would silently accept.

Fitted attributes and ``clone()``
---------------------------------

Fitted state lives in trailing-underscore attributes
(``feature_names_in_``, ``n_features_in_``, ``feature_names_out_``,
``lab_encoder_``), alongside older non-underscore attributes such as
``mapping`` and ``_dim``. Note that ``fit`` re-derives the encoded columns from
``cols``, so ``self.cols`` (and, for the encoders that take a ``mapping``
constructor parameter, ``self.mapping``) are constructor parameters that
``fit`` overwrites with fitted values.

Keep this in mind when combining encoders with ``sklearn.base.clone``:
a clone is a fresh, unfitted estimator — passing a *fitted* encoder as a
constructor argument of another estimator (for example a custom classifier used
with ``cross_val_predict``) results in an unfitted clone in each fold, and
``transform`` on it raises a ``NotFittedError``. Fit the encoder inside the
pipeline or cross-validation loop instead.

``__sklearn_tags__``
--------------------

Encoders implement the scikit-learn tags interface (sklearn >= 1.6). The
supervised/unsupervised split is machine-readable:
``encoder.__sklearn_tags__().target_tags.required`` is ``True`` exactly for the
supervised encoders listed above. The library's tag class additionally carries
``predict_depends_on_y`` (currently always ``False``).

Side by side: ``OneHotEncoder``
-------------------------------

The changelog line "Created a onehot encoder that follows the same conventions
as the rest of the library instead of using sklearns" is where this page's
topic began. The two ``OneHotEncoder`` classes share a name, not a contract:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - ``category_encoders.OneHotEncoder``
     - ``sklearn.preprocessing.OneHotEncoder``
   * - Input
     - DataFrame (or array-like), categories selected by ``cols``
     - array-like, one column per feature
   * - Output
     - DataFrame by default (``return_df``), one binary column per category
     - sparse matrix by default (``sparse_output``), ``dtype`` configurable
   * - Unknown categories
     - ``handle_unknown='value'`` by default (encoded as zeros); ``'error'``,
       ``'return_nan'``, ``'indicator'`` available
     - ``handle_unknown='error'`` by default (raises); ``'ignore'``,
       ``'infrequent_if_exist'`` available
   * - Missing values
     - ``handle_missing='value'`` by default (NaN becomes a countable
       category); ``'error'``, ``'return_nan'``, ``'indicator'``, ``'ignore'``
       available
     - no ``handle_missing`` parameter; NaN is always treated as its own
       category at fit
   * - Column naming
     - ``use_cat_names`` includes category values in output column names
     - ``feature_name_combiner`` customizes names
   * - Category reduction
     - not available
     - ``min_frequency``, ``max_categories``, ``drop``
