import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({
    'A': np.random.randn(100).cumsum(),
    'B': np.random.randn(100).cumsum()
})

df.plot(kind='line', title='Distribution of Income')
plt.show()

x = np.linspace(-2*np.pi, 2*np.pi, 400)
y = np.sin(x) + 0.5*np.sin(2*x)
df.plot(kind='line', title='Distribution of Income')

tips = sns.load_dataset('tips')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)
print(tips.head())
sns.relplot(data=tips, x='total_bill', y='tip', hue='sex', style='time', size='size')
plt.show()

sns.lmplot(data=tips, x='total_bill', y='tip', hue='smoker', height=5, aspect=1.5)
plt.show()

sns.jointplot(data=tips, x='total_bill', y='tip', kind='scatter', hue='sex')
sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex')
sns.jointplot(data=tips, x='total_bill', y='tip', kind='kde')
plt.show()

iris = sns.load_dataset('iris')
sns.pairplot(iris, hue='species', corner=True)
plt.show()

# Compute correlation matrix
corr = iris.drop(columns='species').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title("Feature Correlation")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=tips, x='day', y='total_bill', ax=axes[0])
sns.violinplot(data=tips, x='day', y='total_bill', ax=axes[1])
plt.tight_layout()
plt.show()