import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")

# 1. Data Loading and Cleaning and Display
df = pd.read_csv("yield_df.csv")
df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')
df.drop_duplicates(inplace=True)

print("First five rows of the dataset:")
print(df.head())
print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nDataset shape:", df.shape)
print("\nStatistical summary:")
print(df.describe())

# 2. Data Preparation
cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes',]
cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes',
        'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[cols]

# Separate features (X) and target (y)
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']


# 3. Preprocessing and Pipeline Setup
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

numeric_features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
categorical_features = ['Area', 'Item']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])


# 4. Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# 5. Evaluate the Model
from sklearn.metrics import r2_score, mean_squared_error
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nDecision Tree Regression Results:")
print("R-squared score (Accuracy):", r2)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual hg/ha_yield")
plt.ylabel("Predicted hg/ha_yield")
plt.title("Decision Tree: Actual vs. Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# 6. Predict on New Data
X_new = pd.DataFrame({
    'Area': ['Australia'],
    'Item': ['Wheat'],
    'Year': [2014],                           # Modified from 2013 to 2015
    'average_rain_fall_mm_per_year': [600.0],   # Modified from 534.0 to 600.0
    'pesticides_tonnes': [47000],              # Modified from 45177 to 47000
    'avg_temp': [18.0]                         # Modified from 17.4 to 18.0
})

predicted_yield = pipeline.predict(X_new)
print("\nPrediction for new data:")
print("Predicted yield (in hg/ha):", predicted_yield[0])
