from sklearn.model_selection import GridSearchCV


def grid_search(model, X, y):
    check_model = str(type(model))

    if ("SVC" in check_model):
        print(" ")
        print("Analysing of best parameters for", model, "...")
        param_grid = {'C': [0.1, 1, 10, 100],
                      'gamma': [1, 0.1, 0.01, 0.001],
                      'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

        param_grid = {'C': [0.1], 'gamma': [1], 'kernel': ['rbf']}  # best parameters SVC

    elif ("RandomForestClassifier" in check_model):
        print(" ")
        print("Analysing of best parameters for", model, "...")
        param_grid = {'n_estimators': [10, 100, 120, 200, 300],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'max_depth': [4, 5, 6, 7],
                      'criterion': ['gini', 'entropy']}

        param_grid = {'criterion': ['gini'], 'max_depth': [4], 'max_features': ['sqrt'],
                      'n_estimators': [100]}  # best parameters RFC

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

        param_grid = {'criterion': ['friedman_mse'],  # best parameters GBC
                      'learning_rate': [0.01],
                      'loss': ['deviance'],
                      'max_depth': [3],
                      'max_features': ['log2'],
                      "min_samples_split": [0.1],
                      "min_samples_leaf": [0.1],
                      'n_estimators': [10],
                      'subsample': [0.5]}
    else:
        print("Define grid_param in elif-loop for new model, which you added to models-set")


    '''
    # @todo implement pipeline for upsampling
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    model = Pipeline([
            ('sampling', SMOTE()),
            ('classification', LogisticRegression())
        ])
    
    grid = GridSearchCV(model, params, ...)
    grid.fit(X, y)
    '''


    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
    grid_results = grid.fit(X, y)
    print("Best score for {2}: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_, model))
    params = grid_results.best_params_
    return params
