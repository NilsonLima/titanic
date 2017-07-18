import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import f1_score as F1
from sklearn.metrics import precision_recall_curve

features = []

def extractXy( ):
	global features

	train_frame = pd.read_csv("train/train_modified.csv")
	features = list(filter(lambda x: x not in ["Survived", "PassengerId"], list(train_frame)))

	X = train_frame[features]
	y = train_frame.Survived

	return (X, y)

# def featureImportance( ):
# 	global features
# 	X, y = extractXy( )
#
# 	clf = RandomForestClassifier(oob_score=True, n_estimators=10000)
# 	clf.fit(X, y)
#
# 	feature_importance = clf.feature_importances_
# 	feature_importance = 100 * (feature_importance / feature_importance.max( ))
#
# 	ft_ix = np.where(feature_importance >= 10.0)[0]
#
# 	features = list(map(list(X).__getitem__, ft_ix))
#
# 	return (X[features], y)

def main( ):
	global features

	Xtrain, y = extractXy( )#featureImportance( )
	sqrt = int(math.sqrt(np.shape(Xtrain)[1]))

	# params_dict = {'max_features': range(sqrt-1, np.shape(Xtrain)[1] + 1, 1), 'min_samples_split': range(15, 36, 5), \
	# 'max_depth': range(5, 11, 1), 'n_estimators': range(20, 100, 10), 'min_samples_leaf': range(10, 51, 5)}
	#
	# gridsearch = GridSearchCV(estimator=RandomForestClassifier(warm_start=True, verbose=1, n_jobs=-1), \
	# 						  param_grid=params_dict, n_jobs=-1, cv=10, scoring='accuracy')
	# gridsearch.fit(Xtrain, y)
	# print(gridsearch.best_params_)
	# print(gridsearch.best_score_)
	#
	# forest = gridsearch.best_estimator_
	# y_pred = forest.predict(Xtest)

	test_frame = pd.read_csv("test/test_modified.csv")
	Xtest = test_frame[features]

	forest = RandomForestClassifier(n_estimators=30, min_samples_split=15, min_samples_leaf=10, max_features='sqrt', \
									max_depth=9, warm_start=True, oob_score=True, n_jobs=-1)


	# y_pred = forest.predict(Xtest)
	#
	# data = {'PassengerId': test_frame.PassengerId, 'Survived': y_pred.astype(int)}
	# new_frame = pd.DataFrame(data)
	# new_frame.to_csv("submission.csv", index=False)

	return

if __name__ == '__main__':

	main( )
