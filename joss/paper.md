---
title: 'Category Encoders: a scikit-learn-contrib package of transformers for encoding categorical data'
tags:
  - machine learning
  - python
  - scikit-learn
authors:
 - name: William D McGinnis
   orcid: 0000-0002-3009-9465
   affiliation: "1, 2"
 - name: Chapman Siu
   orcid: 0000-0002-2089-3796
   affiliation: 3
 - name: Andre S
   orcid: 0000-0001-5104-0465
   affiliation: 4
 - name: Hanyu Huang
   orcid:  0000-0001-8503-1014
   affiliation: 5
 - name: Jan Motl
   orcid: 0000-0002-0388-3190
   affiliation: 6
 - name: Kaspar Paul Westenthanner
   orcid: 0009-0007-4284-3953
   affiliation: 7
 - name: Jonathan Castaldo
   orcid: 0009-0001-2553-5920
   affiliation: 8
affiliations:
 - name: Scale Venture Partners
   index: 1
 - name: Helton Labs, LLC
   index: 2
 - name: Suncorp Group Ltd.
   index: 3
 - name: Jungle AI
   index: 4
 - name: Tencent, Inc.
   index: 5
 - name: Czech Technical University in Prague
   index: 6
 - name: Atruvia
   index: 7
 - name: Meta, Inc.
   index: 8
date: 5 December 2017
bibliography: paper.bib
---

# Summary

Category_encoders is a scikit-learn-contrib module of transformers for encoding categorical data. As a scikit-learn-contrib
module, category_encoders is fully compatible with the scikit-learn API [@scikit-learn-api]. It also uses heavily the tools
provided by scikit-learn [@scikit-learn] itself, SciPy [@scipy], pandas [@pandas], and statsmodels [@statsmodels].

The library provides over twenty encoding strategies. These include commonly used encoders such as Ordinal, Hashing, and
OneHot [@idre; @carey; @hashing], as well as contrast-based encoders including Backward Difference, Helmert, Polynomial,
and Sum encoding [@idre; @carey]. It also includes a range of advanced encoders: Target Encoder [@quantile], which uses
the target variable to derive encodings; CatBoost Encoder [@catboost], which applies an ordered variant of target encoding
to reduce overfitting; Weight of Evidence (WOE) Encoder [@woe], widely used in credit scoring and risk modeling;
James-Stein Encoder [@jamesstein], which shrinks estimates toward the overall mean; M-Estimate Encoder [@mestimate];
Generalized Linear Mixed Model (GLMM) Encoder [@glmm]; Count Encoder; Quantile Encoder [@quantile]; Summary Encoder;
Gray Encoder; RankHot Encoder; and BaseN Encoder [@zhang; @onehot; @basen].

# Statement of need

Categorical variables represent a fixed number of possible values, assigning each observation to one of a finite set of
categories. They differ from ordinal variables in that there is no intrinsic ordering among categories. Machine learning
algorithms typically require numeric inputs, so categorical data must be converted into a numeric representation before
modeling. While simple approaches like one-hot encoding are widely available, many real-world datasets contain
high-cardinality categorical features where naive encodings produce excessively wide or uninformative representations.

The original release of category_encoders included a number of commonly used encoders, notably Ordinal, Hashing and OneHot
encoders [@idre][@carey][@hashing], as well as some less frequently used encoders including Backward Difference, Helmert,
Polynomial and Sum encoding [@idre][@carey]. It also included several experimental encoders: LeaveOneOut, Binary and
BaseN [@zhang][@onehot][@basen].

Since then, the library has grown substantially through community contributions.  It now includes over twenty encoding
strategies.  Notable additions include Target Encoder [@quantile], which uses the target variable to derive encodings;
CatBoost Encoder [@catboost], which applies an ordered variant of target encoding to reduce overfitting; Weight of Evidence
(WOE) Encoder [@woe], widely used in credit scoring and risk modeling; James-Stein Encoder [@jamesstein], which shrinks
estimates toward the overall mean; M-Estimate Encoder [@mestimate]; Generalized Linear Mixed Model (GLMM) Encoder [@glmm];
Count Encoder; Quantile Encoder [@quantile]; Summary Encoder; Gray Encoder; and RankHot Encoder.

# State of the field

Several tools exist for encoding categorical variables, but each covers only a subset of available methods. Scikit-learn
provides `OrdinalEncoder` and `OneHotEncoder`, but does not include target-based or statistical encoding methods. The
pandas library offers `get_dummies` for one-hot encoding, but without the fit/transform pattern needed for consistent
train/test handling. In the R ecosystem, the `recipes` and `embed` packages provide some categorical encoding steps, but
with a different API paradigm. The Python libraries Patsy and formulaic support contrast coding schemes but are oriented
toward formula-based model specification rather than general-purpose feature engineering.

Category_encoders fills a gap by consolidating over twenty encoding strategies into a single scikit-learn-compatible
package. It is the only Python library that provides supervised encoding methods such as Target, CatBoost, Weight of
Evidence, James-Stein, and GLMM encoders alongside classical contrast coding and hashing approaches, all with a uniform
interface that handles edge cases like unseen categories and missing values consistently across methods.

# Software design

All encoders in category_encoders inherit from a common `BaseEncoder` class that implements the scikit-learn transformer
interface (`fit`, `transform`, `fit_transform`). This shared base class provides consistent behavior for column selection,
input validation, handling of missing values (via the `handle_missing` parameter), and handling of unseen categories at
transform time (via the `handle_unknown` parameter).

Encoders are divided into two families through mixin classes: `UnsupervisedTransformerMixin` for encoders that operate
solely on feature values (e.g., Ordinal, OneHot, Hashing, BaseN), and `SupervisedTransformerMixin` for encoders that also
use the target variable during fitting (e.g., Target, CatBoost, WOE, James-Stein, GLMM). This design ensures that
supervised encoders properly require a target during `fit` and can optionally use it during `transform`, while maintaining
API compatibility with scikit-learn's `Pipeline` and model selection utilities.

All encoders accept and return pandas DataFrames, automatically detect categorical columns when none are specified, and
support inverse transforms where applicable. The library is designed for production use, with careful handling of edge
cases including previously unseen categories, missing values, and invariant columns.

# Research impact statement

Since its original publication [@onehot], category_encoders has been widely adopted across both academic research and
applied machine learning. The package has been cited in studies spanning diverse domains, demonstrating its utility as
a general-purpose tool for categorical feature engineering.

In machine learning methodology, Poslavskaya and Korolev [-@poslavskaya2023encoding] used category_encoders to conduct a
systematic comparison of encoding methods across OpenML classification benchmarks. In environmental science, Wang et al.
[-@wang2023biochar] applied the library's encoders in data-driven models for biochar-based water treatment. Stojanovic et
al. [-@stojanovic2021fraud] used the package for fraud detection in fintech applications. Badu-Marfo et al.
[-@badumarfo2022composite] employed category_encoders in generative adversarial network models for transportation demand
modeling. Schmitt et al. [-@schmitt2022predicting] used the library to encode categorical process parameters in
pharmaceutical manufacturing prediction models.

The breadth of these applications — from ML benchmarking to environmental science, financial fraud detection,
transportation modeling, and pharmaceutical research — reflects the library's value as a practical, domain-agnostic tool
for working with categorical data in Python.

# AI usage disclosure

AI tools (Claude, Anthropic) were used to assist in drafting and editing the text of this paper. No AI tools were used in
the development of the category_encoders software.

# References
