""" Modified from http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html
"""

from __future__ import print_function

from sklearn import datasets
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from category_encoders.basen import BaseNEncoder
from examples.source_data.loaders import get_mushroom_data
from sklearn.linear_model import LogisticRegression

print(__doc__)

# first we get data from the mushroom dataset
X, y, _ = get_mushroom_data()
X = X.values  # use numpy array not dataframe here
n_samples = X.shape[0]

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# create a pipeline
ppl = Pipeline([
    ('enc', BaseNEncoder(base=2, return_df=False, verbose=True)),
    ('clf', LogisticRegression())
])


# Set the parameters by cross-validation
tuned_parameters = {
    'enc__base': [1, 2, 3, 4, 5, 6]
}

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s\n" % score)
    clf = GridSearchCV(ppl, tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:\n")
    print(clf.best_params_)
    print("\nGrid scores on development set:\n")
    for mean, std, params in clf.grid_scores_:
        print("%s (+/-%s) for %s" % (mean, std * 2, params))

    print("\nDetailed classification report:\n")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))