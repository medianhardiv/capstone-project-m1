from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
	'criterion': ['gini', 'entropy'],
	'max_depth': list(range(2, 10, 2)),
	'min_samples_leaf': [2, 4, 8]
}
tree = DecisionTreeClassifier()
search_cv = GridSearchCV(tree, param_grid, random_state=42)
search_cv.fit(X_train, y_train)