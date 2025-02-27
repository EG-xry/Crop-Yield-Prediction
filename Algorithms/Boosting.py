import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42


# 1. Data Loading and Cleaning
df = pd.read_csv("yield_df.csv")
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop_duplicates(inplace=True)
cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes',
        'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[cols]


# 2. Data Preparation

X = df.drop('hg/ha_yield', axis=1)
y = np.log1p(df['hg/ha_yield'])


# 3. Build Pipeline 
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Area', 'Item'])
    ],
    remainder='passthrough'  
)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('gbr', GradientBoostingRegressor(random_state=RANDOM_STATE))
])

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# 5. Hyperparameter Tuning with RandomizedSearchCV
param_dist = {
    'gbr__n_estimators': [100, 200, 300, 400, 500],
    'gbr__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'gbr__max_depth': [3, 4, 5, 6],
    'gbr__min_samples_split': [2, 5, 10],
    'gbr__min_samples_leaf': [1, 2, 3, 5],
    'gbr__max_features': [None, 'sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=50,           
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

print("\nStarting RandomizedSearchCV for Gradient Boosting Regressor...")
random_search.fit(X_train, y_train)

print("\nBest parameters found:")
print(random_search.best_params_)
print("\nBest cross-validation R² score:", random_search.best_score_)

# 6. Evaluat
best_model = random_search.best_estimator_
y_pred_log = best_model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

r2 = r2_score(y_test_original, y_pred)
mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)

print("\nTest Set Evaluation:")
print("R-squared score:", r2)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual hg/ha_yield")
plt.ylabel("Predicted hg/ha_yield")
plt.title("Improved Gradient Boosting: Actual vs Predicted")
plt.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()], 'r--')
plt.show()

# 7. Predict 
X_new = pd.DataFrame({
    'Year': [2013],
    'average_rain_fall_mm_per_year': [534.0],
    'pesticides_tonnes': [45177.18],
    'avg_temp': [17.4],
    'Area': ['Australia'],
    'Item': ['Wheat']
})

predicted_yield_log = best_model.predict(X_new)
predicted_yield = np.expm1(predicted_yield_log)
print("\nPrediction for new data (Gradient Boosting):")
print("Predicted yield (in hg/ha):", predicted_yield[0])
