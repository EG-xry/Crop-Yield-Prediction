import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
RANDOM_STATE = 1812

# 1. Data Loading and Cleaning
df = pd.read_csv("yield_df.csv")
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop_duplicates(inplace=True)

# 2. Data Preparation
cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes',
        'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[cols]

X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

# 3. Preprocessing Setup
categorical_features = ['Area', 'Item']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# 4. Pipeline & Hyperparameter Search
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=RANDOM_STATE))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

param_dist = {
    'rf__n_estimators': [200, 300, 400, 500, 600],
    'rf__max_depth': [None, 10, 15, 20, 25],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': [10, 'auto', 'sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=100,         
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

print("\nStarting RandomizedSearchCV for Random Forest Regressor...")
random_search.fit(X_train, y_train)

print("\nBest parameters found:")
print(random_search.best_params_)
print("\nBest cross-validation R² score:", random_search.best_score_)

# 5. Evaluate the Best Model on the Test Set
best_model = random_search.best_estimator_
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
plt.title("Random Forest Regression: Actual vs Predicted")
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

predicted_yield = best_model.predict(X_new)
print("\nPrediction for new data (improved RF):")
print("Predicted yield (in hg/ha):", predicted_yield[0])
