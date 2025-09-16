import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("cleaned_dataset.csv")

X = df[['age', 'income', 'age_income_ratio']]
y = df['target']  

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())
