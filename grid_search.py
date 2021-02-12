from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import *



def grid_search(model, X_train, y_train, X_test, y_test):
    check_model = str(type(model))
    model_name = type(model).__name__

    if ("SVC" in check_model):
        print(" ")
        print("Analysing of best parameters for", model, "...")
        param_grid = {'C': [0.1, 1, 10, 100],
                      'gamma': [1, 0.1, 0.01, 0.001],
                      'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

        #param_grid = {'C': [0.1], 'gamma': [0.001], 'kernel': ['poly']}  # best parameters SVC

    elif ("RandomForestClassifier" in check_model):
        print(" ")
        print("Analysing of best parameters for", model, "...")

        param_grid = {'n_estimators': [10, 100, 120, 200, 300],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'max_depth': [4, 5, 6, 7],
                      'criterion': ['gini', 'entropy']}

    elif ("GradientBoostingClassifier" in check_model):
        print(" ")
        print("Analysing of best parameters for", model, "...")
        param_grid = {"loss": ["deviance"],
                      "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                      "min_samples_split": np.linspace(0.1, 0.5, 12),
                      "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                      "max_depth": [3, 5, 8],
                      "max_features": ["log2", "sqrt"],
                      "criterion": ["friedman_mse", "mae"],
                      "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                      "n_estimators": [10, 100]}

    else:
        print("Define grid_param in elif-loop for new model, which you added to models-set")


    model = Pipeline([('sampling', SMOTE(sampling_strategy='minority')), ('clf', model)])
    new_params = {'clf__' + key: param_grid[key] for key in param_grid}

    #grid = GridSearchCV(estimator=model, param_grid=new_params, scoring='accuracy', cv=10, refit=True,
    #                          n_jobs=-1,
    #                          return_train_score=True)

    grid = RandomizedSearchCV(estimator=model, param_distributions=new_params, scoring='accuracy', cv=10, refit=True, n_jobs=-1,
                        return_train_score=True)

    grid_results = grid.fit(X_train, y_train)
    params = grid_results.best_params_
    params = {key.replace('clf__', ''): value  for key, value in params.items()}
    print("Best score for {0}: {1}, using {2}".format(model_name, grid_results.best_score_, params))

    # y_pred = grid_results.predict(X_test)
    # scores = accuracy_score(y_test, y_pred)
    # scores = f1_score(y_test, y_pred)
    # print(scores, "score from grid_search")

    return params