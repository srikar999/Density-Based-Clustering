from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
iris = load_iris()
print(iris)
dbscan = DBSCAN()
dbscan.fit(iris.data)
pca = PCA(n_components=2).fit(iris.data) #finds interreration between diff varaible and makes the process easy
pca_2d = pca.transform(iris.data)
for i in range(0, pca_2d.shape[0]): #returns n. of columns available
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g')
    else:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b')
plt.title('iris data after density based clustering')
plt.show()