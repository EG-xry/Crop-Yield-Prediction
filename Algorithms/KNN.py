import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")

# 1. Load and clean the data
df = pd.read_csv("yield_df.csv")
df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')
df.drop_duplicates(inplace=True)
print("Data Head:\n", df.head())
print("Missing Values:\n", df.isnull().sum())
print("Data Shape:", df.shape)
print("Summary Statistics:\n", df.describe())

# 2. Select the columns to use in the model
cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes',
        'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[cols]
print("\nData after selecting columns:\n", df.head())

# 3. Separate features (X) and target (y)
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

# 4. Set up preprocessing using ColumnTransformer and Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

# Define which columns are numeric and which are categorical
numeric_features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
categorical_features = ['Area', 'Item']

# Create transformers for numeric and categorical data
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

# Combine them into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that first preprocesses the data and then fits the KNN regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('knn', KNeighborsRegressor(n_neighbors=2))
])

# 5. Split the data into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# 7. Evaluate the model on the test set
from sklearn.metrics import r2_score, mean_squared_error
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nKNN Regression Results:")
print("R-squared score (Accuracy):", r2)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual hg/ha_yield")
plt.ylabel("Predicted hg/ha_yield")
plt.title("KNN: Actual vs. Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# 8. Predict on New Data using the pipeline (ensuring proper encoding and scaling)
# Example new data point
X_new = pd.DataFrame({
    'Area': ['Australia'],
    'Item': ['Wheat'],
    'Year': [2013],
    'average_rain_fall_mm_per_year': [534.0],
    'pesticides_tonnes': [45177.18],
    'avg_temp': [17.4]
})

# The pipeline automatically applies the same preprocessing to X_new
predicted_yield = pipeline.predict(X_new)
print("\nPrediction for new data:")
print("Predicted yield (in hg/ha):", predicted_yield[0])
