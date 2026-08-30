import pandas as pd

s = pd.Series([10,20,30,40], index=['a','b','c','d'])
print(s)
print(s['b'])
print(s.iloc[2])

data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'city': ['NYC', 'LA', 'Chicago', 'Houston'],
    'score': [88.5, 92.0, 78.5, 95.5]
}
df = pd.DataFrame(data)
print(df)

df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])
print(df)

print(df.head(2))
print(df.info())
print(df.describe())

ages = df['age']
subset = df[['name', 'score']]
print(ages)
print(subset)

first_row = df.iloc[0]       # first row as Series
first_three = df.iloc[:3]    # first three rows
print(f"first row: {first_row}")
print(f"three rows: {first_three}")

high_scorers = df[df['score'] > 90]
print(high_scorers)

nyc_young = df[(df['city'] == 'NYC') & (df['age'] < 30)]
print(nyc_young)

# Add a new column
df['salary'] = [70000, 80000, 75000, 90000]

# Create column based on existing
df['is_senior'] = df['age'] >= 30

# Remove a column
df = df.drop('salary', axis=1)

# Remove a row
df = df.drop('B', axis=0)

print(df)

print(df.groupby('city')['score'].mean())

# Assignment
data = {
    'student': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'math_score': [85, 92, 78, 88, 95, 70, 82, 90],
    'reading_score': [88, 89, 85, 90, 96, 75, 80, 91],
    'writing_score': [80, 95, 82, 87, 94, 72, 78, 89],
    'gender': ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M']
}
df = pd.DataFrame(data)
print(df.iloc[:3])
print(df.shape)
print(df.columns)
print(df.dtypes)

print(df['math_score'].mean())
print(df['math_score'].min())
print(df['math_score'].max())

print(df['math_score'].describe())
print(df['reading_score'].describe())
print(df['writing_score'].describe())

print(df[df['math_score'] > 85])

print(df[(df['math_score'] > 80) & (df['reading_score'] > 85)])

print(df.loc[df['student'] == 'D'])
print(df.iloc[5:])

df['average_score'] = df[['math_score','reading_score','writing_score']].mean(axis=1)
print(f"Changed DataFrame with mean scores: {df}")