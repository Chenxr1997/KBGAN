from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
import numpy as np

iris = load_iris()
X, y = iris.data[:3], iris.target[:3]
pca = PCA(n_components=2)
X_features1 = pca.fit(X, y).transform(X)
print(X_features1)
X_features2 = SelectKBest(k=1).fit(X, y).transform(X)
print(X_features2)
print(np.hstack((X_features1, X_features2)))

