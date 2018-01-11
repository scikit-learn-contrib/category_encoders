---
title: 'Category Encoders: a scikit-learn-contrib package of transformers for encoding categorical data'
tags:
  - machine learning
  - python
  - sckit-learn
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
affiliations:
 - name: Predikto, Inc.
   index: 1
 - name: Helton Tech, LLC
   index: 2
 - name: Suncorp Group Ltd.
   index: 3
 - name: Jungle AI
   index: 4
 - name: Tencent, Inc.
   index: 5
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

Category_encoders includes a number of pre-existing encoders that are commonly used, notably Ordinal, Hashing and OneHot encoders [@idre][@carey][@hashing]. There are also some
less frequently used encoders including Backward Difference, Helmert, Polynomial and Sum encoding [@idre][@carey]. Finally there are
experimental encoders: LeaveOneOut, Binary and BaseN [@zhang][@onehot][@basen].

The goal of these sorts of transforms is to represent categorical data, which has no true order, as numeric values while
balancing desires to keep the representation in as few dimensions as possible.  Category_encoders seeks to provide access
to the many methodologies for accomplishing such tasks in a simple to use, well tested, and production ready package.


# References

