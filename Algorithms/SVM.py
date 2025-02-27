import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler  # Using RobustScaler for outlier resistance
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

RANDOM_STATE = 42

# 1. Data Loading and Cleaning
df = pd.read_csv("yield_df.csv")
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop_duplicates(inplace=True)

print("First five rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDataset shape:", df.shape)
print("\nStatistical summary:")
print(df.describe())

# 2. Data Preparation
cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[cols]
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']
X_encoded = pd.get_dummies(X, columns=['Area', 'Item'], drop_first=True)
print("\nShape of features after encoding:", X_encoded.shape)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_STATE
)

# 4. Build Pipeline and Grid Search
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('svr', SVR())
])

# Define parameter grids for different kernels.
param_grid = [
    {
        'svr__kernel': ['rbf'],
        'svr__C': [1, 10, 100],
        'svr__gamma': ['scale', 0.1, 0.01],
        'svr__epsilon': [0.1, 0.2]
    },
    # Linear kernel grid
    {
        'svr__kernel': ['linear'],
        'svr__C': [1, 10, 100],
        'svr__epsilon': [0.1, 0.2]
    },
    # Polynomial kernel grid
    {
        'svr__kernel': ['poly'],
        'svr__C': [1, 10, 100],
        'svr__degree': [2, 3],
        'svr__gamma': ['scale', 0.1],
        'svr__epsilon': [0.1, 0.2]
    }
]

# Set up GridSearchCV with 3-fold cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

print("\nStarting grid search...")
grid_search.fit(X_train, y_train)

print("\nBest parameters from GridSearchCV:")
print(grid_search.best_params_)
print("\nBest cross-validation R² score:", grid_search.best_score_)

# 5. Evaluate the Best Model on the Test Set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nTest Set Evaluation:")
print("R-squared score:", r2)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual hg/ha_yield")
plt.ylabel("Predicted hg/ha_yield")
plt.title("SVR: Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# 6. Predict on New Data
X_new = pd.DataFrame({
    'Year': [2015],                           # Adjusted value
    'average_rain_fall_mm_per_year': [600.0],   # Adjusted value
    'pesticides_tonnes': [47000],              # Adjusted value
    'avg_temp': [18.0],                        # Adjusted value
    'Area': ['Australia'],
    'Item': ['Wheat']
})

X_new_encoded = pd.get_dummies(X_new, columns=['Area', 'Item'], drop_first=True)
missing_cols = set(X_encoded.columns) - set(X_new_encoded.columns)
for col in missing_cols:
    X_new_encoded[col] = 0
X_new_encoded = X_new_encoded[X_encoded.columns]  

predicted_yield = best_model.predict(X_new_encoded)
print("\nPrediction for new data (SVR):")
print("Predicted yield (in hg/ha):", predicted_yield[0])
