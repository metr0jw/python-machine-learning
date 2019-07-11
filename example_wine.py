from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn

#k-Neighbors Classifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=4, p=2, weights='uniform')
wine_dataset = load_wine()

#Split train data and test data
X_train, X_test, y_train, y_test = train_test_split(
    wine_dataset['data'], wine_dataset['target'], random_state=0
)
knn.fit(X_train, y_train)

print("wine_dataset's target: \n{}".format(wine_dataset['target']))
print("Size of X_train: {}".format(X_train.shape))
print("Size of y_train: {}".format(y_train.shape))

#Make dataframe by using pandas
#wine dataset from scikit-learn to pandas(matrix)
wine_dataframe = pd.DataFrame(X_train, columns=wine_dataset.feature_names)
pd.plotting.scatter_matrix(wine_dataframe, c=y_train, figsize=(30, 30), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()

#Add new matrix data to data matrix
X_new = np.array([[13.0, 2.34, 2.36, 19.5, 99.7, 2.29, 2.03, 0.36, 1.59, 5.1, 0.96, 2.61, 746]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("prediction: {}".format(prediction))
print("predicted target: {}".format(wine_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("test set prediction:\n {}".format(y_pred))
print("test set accuracy: {:.3f}".format(knn.score(X_test, y_test)))