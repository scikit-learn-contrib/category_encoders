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
provided by scikit-learn [@scikit-learn] itself, scipy[@scipy], pandas[@pandas], and statsmodels[@statsmodels].

Categorical variables (wiki) are those that represent a fixed number of possible values, rather than a continuous number.  Each value assigns the measurement to one of those finite groups, or categories.  They differ from ordinal variables in that the distance from one category to another ought to be equal regardless of the number of categories, as opposed to ordinal variables which have some intrinsic ordering.  As an example:

Ordinal: low, medium, high
Categorical: Georgia, Alabama, South Carolina, … , New York

The machine learning algorithms we will later use tend to want numbers, and not strings, as their inputs so we need some method of coding to convert them.

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

The goal of these sorts of transforms is to represent categorical data, which has no true order, as numeric values while
balancing desires to keep the representation in as few dimensions as possible.  Category_encoders seeks to provide access
to the many methodologies for accomplishing such tasks in a simple to use, well tested, and production ready package.


# References

