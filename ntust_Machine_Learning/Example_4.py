# Example 4_regression 2 variables

from sklearn import linear_model
import matplotlib.pyplot as plt

features = [[0, 1], [1, 3], [2, 8]]
values = [1, 4, 5.5]
reg1 = linear_model.LinearRegression()
reg1.fit(features, values)
print('reg1 coef = ', reg1.coef_)
print('reg1 intercept = ', reg1.intercept_)