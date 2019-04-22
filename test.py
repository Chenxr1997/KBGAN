import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = QuadraticDiscriminantAnalysis()
clf.fit(X, y)

print(clf.predict([[-0.8, -1]]))

