import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("yield_df.csv")
print("Dataset loaded.")
print("Dataset shape:", df.shape) 

num_samples = df.shape[0]
print("Number of samples:", num_samples)


if 'Year' in df.columns:
    sequence_length = df['Year'].nunique()
    print("Sequence length (number of unique years):", sequence_length)
else:
    print("Column 'Year' not found. Define sequence length manually if needed.")
    sequence_length = None  

target_column = "hg/ha_yield" 

categorical_features = ['Area', 'Item']
numerical_features = [col for col in df.columns if col not in categorical_features + [target_column]]

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

print("Data types:")
print(df.dtypes)

if target_column in df.columns:
    target = df[target_column]
    print("\nTarget Distribution Summary:")
    print(target.describe())
    print("Target min:", target.min(), "max:", target.max())

    plt.figure(figsize=(8, 4))
    plt.hist(target, bins=50, edgecolor='k')
    plt.title("Target Distribution: " + target_column)
    plt.xlabel("Yield")
    plt.ylabel("Frequency")
    plt.show()

    target_log = np.log1p(target)
    plt.figure(figsize=(8, 4))
    plt.hist(target_log, bins=50, edgecolor='k', color='orange')
    plt.title("Log-Transformed Target Distribution: log1p(" + target_column + ")")
    plt.xlabel("Log-transformed yield")
    plt.ylabel("Frequency")
    plt.show()
else:
    print(f"Target column '{target_column}' not found in the dataset.")

print("\nSummary of Key Information:")
print("Number of samples:", num_samples)
if sequence_length is not None:
    print("Sequence length (unique years):", sequence_length)
else:
    print("Sequence length: Not determined from data")
print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)
print("Target column:", target_column)
