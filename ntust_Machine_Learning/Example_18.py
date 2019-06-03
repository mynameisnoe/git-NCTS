from sklearn.tree import export_graphviz
from sklearn import tree
import matplotlib.pyplot as plt

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
Y = [0, 0, 1, 1]

col = ['red', 'green']
marker = ['o', '^']
index = 0
while index < len(X):
    type = Y[index]
    plt.scatter(X[index][0], X[index][1], c=col[type], marker=marker[type])
    index += 1
plt.show()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
export_graphviz(clf, out_file="lab18.dot", filled=True, rounded=True,
                special_characters=True)