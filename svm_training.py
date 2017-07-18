import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn import datasets

def extractXy( ):
	train_frame = pd.read_csv("train/train_modified.csv")
	features = list(filter(lambda x: x not in ["Survived", "PassengerId"], list(train_frame)))

	X = train_frame[features]
	y = train_frame.Survived

	return (X, y)

def main( ):
    X, y = extractXy( )

    X = PCA(n_components=3).fit_transform(X)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xs=X[:, 0], ys=X[:, 1], zs=X[:, 2], c=y, cmap=plt.cm.Paired)
    # plt.show( )

    param_grid_svc = [{'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear']}, \
                      {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 10], 'kernel': ['rbf']}, \
                      {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 10], \
                       'kernel': ['poly'], 'degree': [1, 2, 3, 4]}]

    param_grid_bgg = {'n_estimators': range(50, 101, 10)}

    gridsearch = GridSearchCV(SVC( ), param_grid=param_grid_svc, cv=10, n_jobs=-1)
    gridsearch.fit(X, y)

    print(gridsearch.best_estimator_)
    print(gridsearch.best_score_)

    #
    # bagging = BaggingClassifier(base_estimator=gridsearch.best_estimator_, oob_score=True)
    #
    # gridsearch = GridSearchCV(bagging, param_grid=param_grid_bgg, cv=10, n_jobs=-1)
    # gridsearch.fit(X, y)
    #
    # print(gridsearch.best_estimator_)
    # print(gridsearch.best_score_)

    return

if __name__ == '__main__':
    main( )
