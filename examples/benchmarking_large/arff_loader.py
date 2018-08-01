import arff
import numpy as np
import pandas as pd
import requests

# X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
# iris = pd.DataFrame(X, columns=attribute_names)
# iris['class'] = y
#
#
# dataset = openml.download_dataset(61)
# iris = dataset.get_arff()
# iris = pd.DataFrame(iris['data'], columns=[attribute[0] for attribute in iris['attributes']])
# print(iris[:10])




"""
Read data from URL. 

E.g.: arff.load('breast.cancer.arff')

"""
def load(file_name):
    # Load ARFF from web
    # response = requests.get('https://raw.githubusercontent.com/renatopp/arff-datasets/master/classification/' + file_name)
    # html = response.text
    # arff_f = arff.loads(html)

    # Load ARFF from file
    with open('./datasets/arff-datasets-master/classification/' + file_name, 'r') as file:
        arff_f = arff.load(file)

    # ARFF to pandas
    attrs = arff_f['attributes']
    attrs_t = []
    for attr in attrs:
        attrs_t.append(attr[0])
    df = pd.DataFrame(data=arff_f['data'], columns=attrs_t)

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
    else:
        print('Using the last column...', list(df)[-1])
        target = list(df)[-1]

    # Remove rows with a missing target value
    # Justification: They are of no use for strictly supervised learning (semi-supervised learning would still benefit from them)
    df = df.dropna(subset=[target])

    # Get class metadata
    y_unique, y_inversed = np.unique(df[target], return_inverse=True)
    y_counts = np.bincount(y_inversed)

    # Convert the problem into binary classification with {0,1} as class values.
    # Justification: OneHotEncoding and TargetEncoder work only with binary numerical output.
    # Approach: Take a majority class as 1 and the rest as 0.
    majority_class = y_unique[np.argmax(y_counts)]
    df[target] = (df[target]==majority_class).astype('uint8')

    # Preserve only rows with a class value that appears at least twice
    # Justification: It is tough to predict that a testing sample belongs into some class that we have never seen before
    # (one-class classification/outlier detection would be more appropriate)
    # df = df.loc[df[target].isin(y_unique[y_counts>1]),:]
    # y_unique = y_unique[y_counts>1]
    # y_counts = y_counts[y_counts>1]

    # Determine the count of folds that is not going to cause issues.
    # We identify the least common class label and then return min(10, minority_class_count).
    # Justification: If we have only 5 positive samples and 5 negative samples, with stratified cross-validation we can use at best 5 folds.
    y_unique, y_inversed = np.unique(df[target], return_inverse=True)
    y_counts = np.bincount(y_inversed)
    fold_count = min(np.min(y_counts), 10)

    # Target/features split. Encoders expect the target to be in pandas.Series and features in pandas.DataFrame.
    y = df.ix[:, target]
    X = df.drop(target, axis=1)

    # Data type estimation
    for col in X:
        try:
            X[col] = X[col].astype('float', copy=False)
        except ValueError as e:
            pass #print(col, e)

    return X, y, y_unique, fold_count

