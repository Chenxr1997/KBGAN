from sklearn.pipeline import FeatureUnion
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

iris = load_iris()
X, y = iris.data[:3], iris.target[:3]
pca = PCA(n_components=2)
selection = SelectKBest(k=1)
combined_features = FeatureUnion([("pca", pca),("pca2", pca)])
X_features = combined_features.fit(X, y).transform(X)
print(X_features)