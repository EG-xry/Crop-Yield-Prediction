import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("ggplot")
df = pd.read_csv("yield_df.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.drop_duplicates(inplace=True))
print(df.duplicated().sum())
print(df.shape)
print(df.describe())

# Data Visualization
print(len(df['Area'].unique()))
print(len(df['Item'].unique()))


plt.figure(figsize=(15,20))
sns.countplot(y = df['Area'])
plt.show()

plt.figure(figsize=(15,20))
sns.countplot(y = df['Item'])
plt.show()

(df['Area'].value_counts() <400).sum()
country = df['Area'].unique()
yield_per_country = []
for state in country:
    yield_per_country.append(df[df['Area'] == state]['hg/ha_yield'].sum())

df['hg/ha_yield'].sum()
yield_per_country

plt.figure(figsize=(15,20))
sns.barplot(y = country, x = yield_per_country)
plt.show()

crops = df['Item'].unique()
yield_per_crop = []
for crop in crops:
    yield_per_crop.append(df[df['Item'] == crop]['hg/ha_yield'].sum())

plt.figure(figsize=(15,20))
sns.barplot(y = crops, x = yield_per_crop)
plt.show()

df.head()
df.columns
col = ['Year','average_rain_fall_mm_per_year','pesticides_tonnes', 'avg_temp','Area', 'Item', 'hg/ha_yield']
df = df[col]
df.head()
X = df.drop('hg/ha_yield', axis = 1)
y = df['hg/ha_yield']
X.shape
y.shape

import pandas as pd

df = pd.read_csv("yield_df.csv")
print("Dataset loaded.")
print("Dataset shape:", df.shape)

print("Columns:", df.columns)
print("First 5 rows:")
print(df.head())

df_USA = df[df["Area"] == "United States"]
print("\nData from United States:")
print(df_USA)
