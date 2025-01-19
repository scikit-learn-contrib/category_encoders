Contributing Code
=================

Categorical encoding is a python library in the scikit-learn-contrib family for encoding categorical variables as numeric. Our goals are to:

 * Provide a variety of different techniques for encoding
 * Provide a common, familiar API to all of these encoders
 * Support both numpy arrays and pandas dataframes equally
 * Be an all around nice project to use and contribute to
 
With that in mind, we welcome and in fact would love some help.

How to Contribute
=================

The preferred workflow to contribute to git-pandas is:

 1. Fork this repository into your own github account.
 2. Clone the fork and install project via poetry:
    ```
    $ git clone git@github.com:YourLogin/category_encoders.git
    $ cd category_encoders
    $ poetry install
    ```
    
 3. Create a branch for your new awesome feature, do not work in the master branch:
 
    ```
    $ git checkout -b new-awesome-feature
    ```
    
 4. Write some code, or docs, or tests.
 5. When you are done, submit a pull request.
 
Guidelines
==========


 1. Maintain semantics of the scikit-learn API
 2. Write detailed docstrings in numpy format
 3. Support pandas dataframes and numpy arrays as inputs
 4. Write tests

Styleguide: 

We're using ruff for linting. Rules are implemented in the `pyproject.toml` file. To run the linter, use:

    $ poetry run ruff check category_encoders --fix


Running Tests
=============

To run the tests, use:

    $ poetry run pytest tests/
    
Easy Issues / Getting Started
=============================

There are usually some issues in the project github page looking for contributors, if not you're welcome to propose some
 ideas there, or a great first step is often to just use the library, and add to the examples directory. This helps us 
 with documentation, and often helps to find things that would make the library better to use.
