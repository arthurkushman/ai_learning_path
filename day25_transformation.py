import pandas as pd

df1 = pd.DataFrame({'id': [1,2,3], 'name': ['Alice','Bob','Charlie']})
df2 = pd.DataFrame({'id': [4,5], 'name': ['Diana','Eve']})

combined = pd.concat([df1, df2], axis=0, ignore_index=True)
print(combined)

orders = pd.DataFrame({'id': [1,2,2,3], 'product': ['apple','banana','orange','grape'], 'quantity': [2,1,3,2]})
# Outer join (all records from both)
outer = pd.merge(combined, orders, on='id', how='outer')
print(outer)

wide = pd.DataFrame({
    'student': ['S1','S2','S3'],
    'math': [90,85,88],
    'science': [92,88,91],
    'english': [85,90,87]
})

long = pd.melt(wide, id_vars=['student'], var_name='subject', value_name='score')
print(long)

pivoted = long.pivot(index='student', columns='subject', values='score')
print(pivoted)

sales = pd.DataFrame({
    'region': ['North','South','North','South','North'],
    'month': ['Jan','Jan','Feb','Feb','Mar'],
    'sales': [150, 200, 180, 220, 210]
})

region_sales = sales.groupby(['region'])['sales'].sum()
print(region_sales)

avg_sales_per_month = sales.groupby(['month'])['sales'].mean()
print(avg_sales_per_month)

sum_of_sales = sales.groupby(['region', 'month'])['sales'].sum()
print(sum_of_sales)

agg_sales = sales.groupby(['region'])['sales'].agg(['mean', 'min', 'max'])
print(agg_sales)

pivot_tbl = pd.pivot_table(sales, values='sales', index='region', columns='month', aggfunc="sum", fill_value=0)
print(pivot_tbl)

students = pd.DataFrame({
    'grade': ['A','B','A','C','B','A','C','B'],
    'passed': [True, True, False, True, True, False, False, True]
})

ct = pd.crosstab(students['grade'], students['passed'], normalize='index')
print(ct)