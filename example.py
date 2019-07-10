from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn

knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')
iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)
knn.fit(X_train, y_train)

print("iris_dataset's target: \n{}".format(iris_dataset['target']))
print("Size of X_train: {}".format(X_train.shape))
print("Size of y_train: {}".format(y_train.shape))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)#pd.plotting.scatter_matrix(wine_dataframe, c=y_train, figsize=(35, 35), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()

X_new = np.array([[4.3, 2.0, 2.0, 3.0]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("prediction: {}".format(prediction))
print("predicted target: {}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("test set prediction:\n {}".format(y_pred))
print("test set accuracy: {:.3f}".format(knn.score(X_test, y_test)))