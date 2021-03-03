import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:\dataArc\Advertising.csv")

X = df.drop('Sales', axis=1)
y = df[["Sales"]]

fig, axs = plt.subplots(1, 3, sharey=True)
df.plot(kind="scatter", x='TV', y='Sales', ax=axs[0], figsize=(10, 5))
df.plot(kind="scatter", x='Radio', y='Sales', ax=axs[1], figsize=(10, 5))
df.plot(kind="scatter", x='Newspaper', y='Sales', ax=axs[2], figsize=(10, 5))
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
print(reg_model.intercept_)
print(reg_model.coef_)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

reg_model.score(X_train, y_train)

y_pred = reg_model.predict(X_test)
print("mse: ", mean_squared_error(y_test, y_pred))
print("rmse: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("mae: ",  mean_absolute_error(y_test, y_pred))
