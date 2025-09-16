import pandas as pd
import numpy as np

df = pd.read_csv("example_dataset.csv")

df['age'].fillna(df['age'].mean(), inplace=True)
df['income'] = pd.to_numeric(df['income'], errors='coerce')
df['income'].fillna(df['income'].median(), inplace=True)

q1 = df['income'].quantile(0.25)
q3 = df['income'].quantile(0.75)
iqr = q3 - q1
df = df[(df['income'] >= q1 - 1.5*iqr) & (df['income'] <= q3 + 1.5*iqr)]

df['age_income_ratio'] = df['age'] / df['income']

df.to_csv("cleaned_dataset.csv", index=False)

print("Data cleaning complete. Saved as cleaned_dataset.csv")
