import pandas as pd
import numpy as np
from scipy import stats

data = {
    'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'David', 'Eve', 'Frank', 'Grace', 'Heidi'],
    'age': [25, 30, 25, 35, 30, 40, 28, 33, None, 29],
    'income': [50000, 60000, 50000, 80000, 60000, 95000, 45000, 70000, 55000, None],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', ' NYC ', 'Houston', 'Chicago', 'LA', 'Miami'],
    'gender': ['F', 'M', 'F', 'M', 'M', 'M', 'F', 'M', 'F', 'F'],
    'score': [85, 92, 85, 78, 92, 88, 95, 80, 90, 87]
}
df = pd.DataFrame(data)

print(df['city'].replace({'NYC': 'New York', 'LA': 'Los Angeles'}))
df['age'] = df['age'].fillna(df['age'].median())
df['income'] = df['income'].fillna(df['income'].mean())
print(df)

df_clean = df.drop_duplicates()
print(df_clean)

Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['income'] < lower_bound) | (df['income'] > upper_bound)]
print(outliers)

df_clean = df[(df['income'] >= lower_bound) & (df['income'] <= upper_bound)]
print(df_clean)

# df['age_group'] = df['age'].cut({'young': df['income'] < 30})
# print(df['age_group'])

z_scores = np.abs(stats.zscore(df['income']))
df_clean = df[z_scores < 3]
print(df_clean)

print(df.isnull().sum())
df.to_csv('cleaned_data.csv', index=False)
