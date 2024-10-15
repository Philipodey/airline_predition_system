import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv(r"C:\Users\DELL\Downloads\airline_flight\flight_dataset.csv")

pd.set_option("display.max_columns", None)

train_data.head()

train_data.info()

df = pd.read_csv(r"C:\Users\DELL\Downloads\airline_flight\flight_dataset.csv", index_col=0)

df.info()

df.isnull().sum()

print(df.shape)

df['airline'].value_counts()

df.destination_city.value_counts()

df['source_city'].value_counts()

# Preprocessing

print(df.columns)

print(df)

df = df.drop("flight", axis=1)

df['class'] = df['class'].apply(lambda x: 1 if x == "Business" else 0)

df.stops = pd.factorize(df.stops)[0]
print(df)

# from the graph we can see that AirAsia Airways have the highest Price

sns.barplot(y="price", x="airline", data=df.sort_values("price", ascending=False));

df.describe()


def one_hot_encode_column(data_frame, column_name):

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)

    # Select the column for encoding
    column_df = df[[column_name]]

    # Perform one-hot encoding
    encoded = encoder.fit_transform(column_df)

    # Convert the encoded data into a DataFrame with appropriate column names
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column_name]))

    # Concatenate the original DataFrame with the new encoded columns
    df_encoded = pd.concat([df, encoded_df], axis=1)

    # Drop the original column that was encoded
    df_encoded.drop(column_name, axis=1, inplace=True)

    return df_encoded


df = one_hot_encode_column(df, "airline")
print(df.head())

df['source_city'].value_counts()

# source_city vs price
sns.barplot(y="price", x="source_city", data=df.sort_values("price", ascending=False))

source_city = df[["source_city"]]

df = one_hot_encode_column(df, 'source_city')
print(df.columns)

destination = df[["destination_city"]]

df = one_hot_encode_column(df, 'destination_city')

print(df[['departure_time']])
df = one_hot_encode_column(df, 'departure_time')
print(df.head())

print(df.arrival_time)
df = one_hot_encode_column(df, 'arrival_time')
print(df.head())

# Training Regression Model
X, y = df.drop('price', axis=1), df.price

print("x data", X)
print("y data", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_jobs=-1)
model.fit(X_train, y_train)

model.score(X_test, y_test)
print(model.score(X_test, y_test))

# Hyperparameter tuning

y_pred = model.predict(X_test)

print("Root Squared: ", r2_score(y_test, y_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("Root Mean Squares Error: ", math.sqrt(mean_squared_error(y_test, y_pred)))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Flight price")
plt.ylabel("Predicted Flight Price")
plt.ylabel("Predicted VS Actual Price")

print(df["price"].describe())

importance = dict(zip(model.feature_names_in_, model.feature_importances_))
sorted_importances = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print(sorted_importances)


model = RandomForestRegressor(n_jobs=-1)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features" : ["auto", "sqrt"]
}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_