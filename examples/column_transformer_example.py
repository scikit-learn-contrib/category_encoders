from examples.source_data.loaders import get_mushroom_data
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder

# get data from the mushroom dataset
X, y, _ = get_mushroom_data()

# encode the specified columns
ct = ColumnTransformer(
    [
        ('Target encoding', TargetEncoder(), ['bruises', 'odor'])
    ], remainder='passthrough'
)
encoded = ct.fit_transform(X=X, y=y)

# show the result
print(encoded)
