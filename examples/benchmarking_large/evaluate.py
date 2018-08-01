import numpy as np
import sklearn
import sklearn.metrics.scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler, Imputer
# from sklearn.compose import ColumnTransformer, make_column_transformer
# from numba import jit, int32


class Columns(BaseEstimator, TransformerMixin):
    """
    A transformer, which selects only some of the columns based on their names.
    Warning: We assume that the input is a DataFrame.
    """

    def __init__(self, names=None):
        self.names = names

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X[self.names]


# @jit
def evaluate(X, y, fold_count, encoder, model, class_values):
    # generate some data to play with
    # X, y = samples_generator.make_classification(n_samples=1000, n_informative=5, n_classes=3, random_state=42)

    # Define the pipeline.
    # Scaler does not work with missing values
    # encoder = category_encoders.OneHotEncoder()
    # model = svm.SVC(kernel='linear', probability=True)

    numeric = list(X.select_dtypes(include=[np.number]).columns.values)
    categorical = list(X.select_dtypes(include='object').columns.values)

    if len(numeric)==0:
        pipeline = Pipeline([
            ("features", FeatureUnion([
                ('categorical', make_pipeline(Columns(names=categorical), encoder))
            ])),
            ('model', model)
        ], memory='./transformers/')
    else:
        pipeline = Pipeline([
            ("features", FeatureUnion([
                ('numeric', make_pipeline(Columns(names=numeric), Imputer(), StandardScaler())),
                ('categorical', make_pipeline(Columns(names=categorical), encoder))
            ])),
            ('model', model)
        ], memory='./transformers/')

    # For debugging of the transformers only
    # print("DEBUG IN EVALUATE")
    # pipeline2 = Pipeline([
    #     ("features", FeatureUnion([
    #         # ('numeric', make_pipeline(Columns(names=numeric), Imputer(copy=False, verbose=True), StandardScaler(copy=False))),
    #         ('categorical', make_pipeline(Columns(names=categorical), encoder))
    #     ]))
    # ], memory='./transformers/')
    # X_transformed = pipeline2.fit_transform(X, y)
    # print('ok', np.any(np.isnan(X_transformed)))

    # Make a dictionary of the scorers.
    # We have to say which measures require predicted labels and which probabilities.
    # Also, we have to pass all the unique class values, otherwise measures may complain during the cross-validation. Only accuracy does not accept labels parameter.
    # Also, we have to say how to average scores in F-measure.
    log_loss_score = sklearn.metrics.make_scorer(sklearn.metrics.log_loss, needs_proba=True, labels=class_values)
    accuracy_score = sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)
    f1_macro = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average='macro', labels=class_values)
    score_dict = {'accuracy':accuracy_score, 'f1_macro':f1_macro, 'log_loss':log_loss_score} # fails because the training data do not contain all the labels


    # Perform cross-validation
    # It returns a dict containing training scores, fit-times and score-times in addition to the test score.
    scores = cross_validate(estimator=pipeline, X=X, y=y, scoring=score_dict, cv=StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=2001), return_train_score=True)

    return scores
