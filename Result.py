import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = {
    'Algorithm': [
        'Linear Regression', 
        'KNN', 
        'Decision Tree', 
        'Random Forest', 
        'SVM', 
        'Gradient Boosting Tree'
    ],
    'r2': [
        0.7486,    # Linear Regression
        0.9859,    # KNN
        0.9772,    # Decision Tree
        0.9795,    # Random Forest
        0.5565,    # SVM
        0.9694     # Gradient Boosting Tree (Boosting)
    ],
    'Prediction': [
        81051,      # Linear Regression
        19560,      # KNN
        20301,      # Decision Tree
        21697,      # Random Forest
        81360.98,   # SVM
        17599       # Gradient Boosting Tree
    ]
}

df = pd.DataFrame(data)
actual_prediction = 17609
df['Diff'] = abs(df['Prediction'] - actual_prediction)

# Theoretical Ranking
# Random Forest > Decision Tree > KNN > Boosting > SVM > Linear Regression
theoretical_order = [
    'Random Forest', 
    'Decision Tree', 
    'KNN', 
    'Gradient Boosting Tree',  
    'SVM', 
    'Linear Regression'
]
df['Theoretical'] = df['Algorithm'].apply(lambda x: theoretical_order.index(x) + 1)

# Ranking Each Metric and Overall Ranking
# For r², higher is better: rank in descending order.
df['r2_rank'] = df['r2'].rank(ascending=False)
# For prediction difference, lower is better: rank in ascending order.
df['diff_rank'] = df['Diff'].rank(ascending=True)
# Compute an average overall ranking from the three metrics.
df['Overall_rank'] = (df['r2_rank'] + df['diff_rank'] + df['Theoretical']) / 3
norm = plt.Normalize(df['Overall_rank'].min(), df['Overall_rank'].max())
colors = plt.cm.RdYlGn_r(norm(df['Overall_rank']))

# Graph 1: r² Score
plt.figure(figsize=(10, 6))
plt.barh(df['Algorithm'], df['r2'], color=colors)
plt.xlabel('r² Score')
plt.title('Model r² (Higher is Better)')
for i, v in enumerate(df['r2']):
    plt.text(v + 0.005, i, f"{v:.4f}", va='center', clip_on=False)
plt.gca().margins(x=0.05)
# Remove the right spine
ax = plt.gca()
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

# Graph 2: Absolute Prediction Difference (Logarithmic x-axis)
plt.figure(figsize=(10, 6))
plt.barh(df['Algorithm'], df['Diff'], color=colors)
plt.xlabel('Absolute Difference from 17609 (log scale)')
plt.title('Prediction Accuracy (Lower is Better)')
plt.xscale('log')
max_diff = df['Diff'].max()
plt.xlim(1, max_diff * 1.5)
for i, v in enumerate(df['Diff']):
    plt.text(v * 1.1, i, f"{v:.0f}", va='center', clip_on=False)
plt.gca().margins(x=0.1)
ax = plt.gca()
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

# Graph 3: Theoretical Ranking
plt.figure(figsize=(10, 6))
plt.barh(df['Algorithm'], df['Theoretical'], color=colors)
plt.xlabel('Theoretical Rank (Lower is Better)')
plt.title('Theoretical Ranking')
for i, v in enumerate(df['Theoretical']):
    plt.text(v + 0.1, i, f"{int(v)}", va='center', clip_on=False)
plt.gca().margins(x=0.1)
ax = plt.gca()
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()
