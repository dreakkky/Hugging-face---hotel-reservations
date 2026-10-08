import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

# Login using e.g. `huggingface-cli login` to access this dataset
dataset = pd.read_csv("hf://datasets/aki-008/hotel_data/hotel_bookings.csv")

print(dataset.head())
print(dataset.tail())
print(dataset.dtypes)
print(dataset.isnull().sum()) # missing values

leaky_columns = ["reservation_status", "reservation_status_date"]
X, y = dataset.drop(columns=["is_canceled"] + leaky_columns), dataset["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

numerical_cols = []
catagorical_col = []
for col in X_train:
    if X_train[col].dtypes == object:
        catagorical_col.append(col)
    if X_train[col].dtypes == int or X_train[col].dtypes == float:
        numerical_cols.append(col)

print(numerical_cols)
print(catagorical_col)

numerical_data = SimpleImputer(strategy="mean")
catagorical_data = Pipeline(steps=[
    ('impute', SimpleImputer(strategy="most_frequent")),
    ('encode', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

transformer = ColumnTransformer(transformers=[
    ('cat', catagorical_data, catagorical_col),
    ('num', numerical_data, numerical_cols)
])

model = Pipeline(steps=[
    ('transformer', transformer),
    ('model', RandomForestClassifier(random_state=1))
])

model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(f"prediction: {prediction}")

mae = mean_absolute_error(y_test, prediction)
print(f"mae: {mae:.5f}")
