import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import datasets

def main( ):
	iris = datasets.load_iris( )
	X = iris.data
	y = iris.target

	#parameters dictionary for GridSearchCV operation
	params_dict = {'max_features': range(2, 5, 1), 'min_samples_split': range(5, 31, 5),\
	'max_leaf_nodes': range(5, 11, 1), 'n_estimators': [200]}

	clf = RandomForestClassifier(oob_score=True)
	gridsearch = GridSearchCV(estimator=clf, param_grid=params_dict, cv=3, n_jobs=-1)
	gridsearch.fit(X, y)

	print(gridsearch.best_estimator_)
	print(gridsearch.best_score_)
	print(gridsearch.best_params_)

	# #cross validation for 
	# forest = RandomForestClassifier(oob_score=True, n_estimators=50, min_samples_split=5, max_features=2, max_leaf_nodes=5)
	# cv = cross_val_score(forest, X, y, n_jobs=-1, cv=3)
	# print(np.mean(cv))

	return

if __name__ == '__main__':
	main( )