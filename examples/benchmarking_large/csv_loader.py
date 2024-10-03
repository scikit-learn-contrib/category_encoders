import numpy as np
import pandas as pd

"""
Read data in arff format from URL. 

E.g.: csv_loader.load('car.csv')

"""


def load(file_name):
    # Load CSV from file
    df = pd.read_csv('./datasets/article/' + file_name, sep=None, na_values='?')

    # Target column estimation
    if 'class' in list(df):
        target = 'class'
    elif 'Class' in list(df):
        target = 'Class'
    elif 'type' in list(df):
        target = 'type'
    elif 'TYPE' in list(df):
        target = 'TYPE'
    elif 'Type' in list(df):
        target = 'Type'
    elif 'symboling' in list(df):
        target = 'symboling'
    elif 'OVERALL_DIAGNOSIS' in list(df):
        target = 'OVERALL_DIAGNOSIS'
    elif 'LRS-class' in list(df):
        target = 'LRS-class'
    elif 'num' in list(df):
        target = 'num'
    elif 'Class_attribute' in list(df):
        target = 'Class_attribute'
    elif 'Contraceptive_method_used' in list(df):
        target = 'Contraceptive_method_used'
    elif 'surgical_lesion' in list(df):
        target = 'surgical_lesion'
    elif 'band_type' in list(df):
        target = 'band_type'
    elif 'Survival_status' in list(df):
        target = 'Survival_status'
    elif 'surgical lesion' in list(df):
        target = 'surgical lesion'
    elif 'decision' in list(df):
        target = 'decision'
    elif 'ACTION' in list(df):
        target = 'ACTION'
    elif 'IsBadBuy' in list(df):
        target = 'IsBadBuy'
    elif 'TEACHER_FIRST_VISIT_FLAG' in list(df):
        target = 'TEACHER_FIRST_VISIT_FLAG'
    elif 'Community' in list(df):
        target = 'Community'
    elif 'shot_made_flag' in list(df):
        target = 'shot_made_flag'
    elif 'presence_of_spiders' in list(df):
        target = 'presence_of_spiders'
    elif 'Delay' in list(df):
        target = 'Delay'
    elif 'WnvPresent' in list(df):
        target = 'WnvPresent'
    elif 'Community Membership_Family' in list(df):
        target = 'Community Membership_Family'
    elif 'sex' in list(df):
        target = 'sex'
    else:
        print('Using the last column...', list(df)[-1])
        target = list(df)[-1]

    # Remove rows with a missing target value
    # Justification: They are of no use for strictly supervised learning (semi-supervised learning would still benefit from them)
    df = df.dropna(subset=[target]).reset_index(drop=True)

    # Get class metadata
    y_unique, y_inversed = np.unique(df[target], return_inverse=True)
    y_counts = np.bincount(y_inversed)

    # Convert the problem into binary classification with {0,1} as class values.
    # Justification: OneHotEncoding and TargetEncoder work only with binary numerical output.
    # Approach: Take a majority class as 1 and the rest as 0.
    majority_class = y_unique[np.argmax(y_counts)]
    df[target] = (df[target] == majority_class).astype('uint8')

    # Determine the count of folds that is not going to cause issues.
    # We identify the least common class label and then return min(10, minority_class_count).
    # Justification: If we have only 5 positive samples and 5 negative samples, we can use at best 5 folds with stratified cross-validation.
    y_unique, y_inversed = np.unique(df[target], return_inverse=True)
    y_counts = np.bincount(y_inversed)
    fold_count = min(np.min(y_counts), 10)

    # Target/features split. Encoders expect the target to be in pandas.Series and features in pandas.DataFrame.
    y = df.loc[:, target]
    X = df.drop(target, axis=1)

    # Estimate, which columns are nominal. If there is no string column in the data, assume that all integers are nominal.
    nominal_columns = [key for key, value in X.dtypes.items() if ('object' in value.name)]
    if len(nominal_columns) == 0:
        nominal_columns = [key for key, value in X.dtypes.items() if ('int' in value.name)]

    # Data type estimation
    for col in X:
        try:
            X[col] = X[col].astype('float', copy=False)
        except ValueError:
            pass

    return X, y, fold_count, nominal_columns
