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

Running Mutation Tests
======================

Mutation testing (mutmut 3.x) is configured in `pyproject.toml`
(`[tool.mutmut]`): the library source under `category_encoders/` is mutated
and the suite under `tests/` runs against each mutant. A whole-package
campaign is far too slow for routine use (~30 h on 8 workers, measured), so
run it scoped to a single module while working on that module: narrow
`source_paths` in `pyproject.toml` to that one file, add `also_copy` so the
rest of the package is still importable inside mutmut's sandbox (mutmut only
copies files matched by `source_paths`), run the campaign, and restore the
wider setting afterwards:

    # temporarily:
    #   source_paths = ["category_encoders/basen.py"]
    #   also_copy = ["category_encoders"]
    $ poetry run mutmut run --max-children 8

    # Inspect the survivors and their diffs
    $ poetry run mutmut results
    $ poetry run mutmut show <mutant>

Use it as a test-quality check for a module you just changed: any surviving
mutant marks behavior that the current tests do not pin down. `wrapper.py` is
excluded from mutation (`do_not_mutate_patterns` in `pyproject.toml`) because
its tests spawn process pools and nested cross-validation fits that hang
under forked mutants.

Easy Issues / Getting Started
=============================

There are usually some issues in the project github page looking for contributors, if not you're welcome to propose some
 ideas there, or a great first step is often to just use the library, and add to the examples directory. This helps us 
 with documentation, and often helps to find things that would make the library better to use.
