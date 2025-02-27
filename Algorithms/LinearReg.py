import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# 1. Read and clean the data
df = pd.read_csv("yield_df.csv")
df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')

print(df.head())
print(df.isnull().sum())
print("Number of duplicates before dropping:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Number of duplicates after dropping:", df.duplicated().sum())
print("Data shape:", df.shape)
print(df.describe())

# 2. Select columns and reorder (if needed)
cols = [
    'Year',
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp',
    'Area',
    'Item',
    'hg/ha_yield'
]
df = df[cols]
print("\nFirst 5 rows after reordering columns:")
print(df.head())

# 3. Define features (X) and target (y)
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

# 4. One-hot encode 'Area' and 'Item' manually for your training data
X_encoded = pd.get_dummies(X, columns=['Area', 'Item'], drop_first=True)
print("\nShape of features after encoding:", X_encoded.shape)
print("Shape of target:", y.shape)

# 5. Train a Linear Regression Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate on the test set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nEvaluation on Test Set:")
print("R-squared score (Accuracy):", r2)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Optional: Plot Actual vs. Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual hg/ha_yield")
plt.ylabel("Predicted hg/ha_yield")
plt.title("Linear Regression: Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# 7. Predict on New Data Manually (no pipeline)
# Example input
X_new = pd.DataFrame({
    'Area': ['Australia'],
    'Item': ['Wheat'],
    'Year': [2013],
    'average_rain_fall_mm_per_year': [534.0],
    'pesticides_tonnes': [45177.18],
    'avg_temp': [17.4]
})

# One-hot encode X_new with the same approach (drop_first=True)
X_new_encoded = pd.get_dummies(X_new, columns=['Area', 'Item'], drop_first=True)

# Align columns with X_encoded: if any columns are missing, add them as zeros in one step.
missing_cols = set(X_encoded.columns) - set(X_new_encoded.columns)
if missing_cols:
    missing_df = pd.DataFrame(0, index=X_new_encoded.index, columns=list(missing_cols))
    X_new_encoded = pd.concat([X_new_encoded, missing_df], axis=1)

# Reorder columns to match exactly the training set's encoded features
X_new_encoded = X_new_encoded[X_encoded.columns]

# Finally, predict the yield for the new data
predicted_yield = model.predict(X_new_encoded)
print("\nPrediction for new data:")
print("Predicted yield (in hg/ha):", predicted_yield[0])
