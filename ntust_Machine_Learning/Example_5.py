# Example 5_regression_score

from sklearn import linear_model

reg1 = linear_model.LinearRegression()
reg1.fit([[0, 0], [1, 1], [2, 2]], [1, 4, 8])
print("coef =", reg1.coef_)
print("intercept =", reg1.intercept_)
print("predict =", reg1.predict([[0.5, 0.5], [1.5, 1.5], [2, 1], [1, 2]]))

result1 = reg1.predict([[0.5, 0.5], [1.5, 1.5], [2, 1], [1, 2]])
print("score 1 =", reg1.score([[0, 0], [1, 1], [2, 2]], [1, 4, 8]))
print("score 2 =", reg1.score([[0, 0], [1, 1], [2, 2]], [1, 2, 3]))
