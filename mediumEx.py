import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv("D:\dataArc\SalaryData.csv")

X = df[["YearsExperience"]]
y = df[["Salary"]]

plt.scatter(X, y)
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

reg_model = LinearRegression().fit(X, y)
print("B0: ", reg_model.intercept_[0])
print("B1: ", reg_model.coef_[0][0])
print("y^: ", reg_model.intercept_[0] + reg_model.coef_[0][0] * 6)

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9}, ci=False, color="r")
g.set_title(f"Model Denklemi: Salary = {round(reg_model.intercept_[0],2)} + {round(reg_model.coef_[0][0], 2)} * YearsExperience")
g.set_xlabel("YearsExperience")
g.set_ylabel("Salary")
plt.show()

