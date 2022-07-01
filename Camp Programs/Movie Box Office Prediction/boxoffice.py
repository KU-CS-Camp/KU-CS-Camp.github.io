import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pandas.read_csv('movie_cost_revenue.csv')
data.describe()

X = DataFrame(data, columns=['production_budget_usd'])
y = DataFrame(data, columns=['worldwide_gross_usd'])
print(X.shape)

plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha=0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()
plt.clf()

regression = LinearRegression()
regression.fit(X, y)

print(regression.coef_)
print(regression.intercept_)

predict_list = regression.predict(X)

plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha=0.3)

plt.plot(X['production_budget_usd'], predict_list, color='red', linewidth=3)

plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()

regression.score(X, y)
