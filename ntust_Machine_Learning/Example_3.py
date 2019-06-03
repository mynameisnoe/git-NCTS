# Example 3
from sklearn import linear_model
import matplotlib.pyplot as plt

features = [[1], [2], [3]]
values = [1, 4, 5.5]
plt.scatter(features, values, c='green')

regl = linear_model.LinearRegression()
regl.fit(features, values)

print('coef=%f, intercept=%f' %(regl.coef_, regl.intercept_))
range1 = [-1, 3]
plt.plot(range1, regl.coef_ * range1 + regl.intercept_)
