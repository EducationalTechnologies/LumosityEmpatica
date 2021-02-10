import data_helper
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from feature_extraction import extract_df_with_features
from feature_selection import select_features
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV


def get_models():
    models = dict()
    models['support_vector_machines'] = SVC(C=0.1, gamma=1, kernel='rbf')
    models['random_forest'] = RandomForestClassifier(criterion='gini', max_depth=4,max_features='sqrt', n_estimators=100)
    models['gradient_boosting'] = GradientBoostingClassifier(criterion='friedman_mse',
                                                             learning_rate=0.01,
                                                             loss='deviance',
                                                             max_depth=3,
                                                             max_features='log2',
                                                             min_samples_split=0.1,
                                                             min_samples_leaf=0.1,
                                                             n_estimators=10,
                                                             subsample=0.5)
    return models


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    #for k, v in params.items():
    #    model.set_params(**{k: v})
    model_name = type(model).__name__
    print(" ")
    print("Training model {0}: ".format(model_name))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = accuracy_score(y_test, y_pred)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.title('Accuracy Score with {1} model: {0} '.format(scores, model_name), fontsize=10)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('Confusion_matrix_and_ROC_Curve/confusion_matrix {0}.png'.format(model_name))
    plt.show()
    return scores

'''
def grid_search(model, X, y):
    check_model = str(type(model))

    if ("SVC" in check_model):
        print(" ")
        print("Analysing of best parameters for", model, "...")
        param_grid = {'C': [0.1, 1, 10, 100],
                      'gamma': [1, 0.1, 0.01, 0.001],
                      'kernel': ['linear','rbf', 'poly', 'sigmoid']}

        param_grid = {'C': [0.1], 'gamma': [1], 'kernel': ['rbf']}  #best parameters SVC

    elif ("RandomForestClassifier" in check_model):
        print(" ")
        print("Analysing of best parameters for", model, "...")
        param_grid = {'n_estimators': [10, 100, 120, 200, 300],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'max_depth': [4, 5, 6, 7],
                      'criterion': ['gini', 'entropy']}

        param_grid = {'criterion': ['gini'], 'max_depth': [4], 'max_features': ['sqrt'], 'n_estimators': [100]} #best parameters RFC

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

        param_grid = {'criterion': ['friedman_mse'],  #best parameters GBC
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

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
    grid_results = grid.fit(X, y)
    print("Best score for {2}: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_, model))
    params = grid_results.best_params_
    return params
'''

# creating ROC curves for all models with using roc_auc_score
def ROC_curve(models):

    classifiers = list(models.values())
    # classifiers = [SVC(C=0.1, gamma=1, kernel='rbf'),
    #               RandomForestClassifier(criterion='gini', max_depth=4, max_features='sqrt', n_estimators=100),
    #               GradientBoostingClassifier(criterion='friedman_mse', learning_rate=0.01, loss='deviance',
    #                                          max_depth=3, max_features='log2', min_samples_split=0.1,
    #                                          min_samples_leaf=0.1, n_estimators=10, subsample=0.5)]

    table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
    for cls in classifiers:
        cls.probability = True
        model = cls.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[::, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        table = table.append({'classifiers': cls.__class__.__name__, 'fpr': fpr,
                              'tpr': tpr, 'auc': auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    table.set_index('classifiers', inplace=True)
    for i in table.index:
        plt.plot(table.loc[i]['fpr'],
                 table.loc[i]['tpr'], label="{}, AUC={:.3f}".format(i, table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title('ROC Curves', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 9}, loc='lower right')
    plt.savefig('confusion_matrix_and_ROC_Curve/ROC_Curve.png', dpi=400)
    plt.show()


if __name__ == "__main__":
    data_folder = "manual_sessions/lumosity-dataset"
    ignore_files = []
    to_exclude = ['ACC', 'OenName', 'RecordingID', 'ApplicationName']
    target_classes = ['mistake']
    tensor_data, annotations, attributes = data_helper.get_data_from_files(data_folder, ignore_files=ignore_files,
                                                                           res_rate=25,
                                                                           to_exclude=to_exclude)
    print("Shape of the tensor_data is: " + str(np.shape(tensor_data)))
    print("Shape of the annotation is: " + str(np.shape(annotations)) + "\n")

    attributes = ['bvp', 'gsr', 'hrv', 'ibi', 'tmp']

    # split the dataset between train and test
    X_train, y_train, X_test, y_test = data_helper.split_data_train_test(tensor_data, annotations,
                                                                         train_test_ratio=0.85, random_shuffling=True)
    # feature extraction both on the training set and the test set
    X_train = extract_df_with_features(X_train, y_train, attributes, target_classes, data_folder, is_train_set=True)
    X_test = extract_df_with_features(X_test, y_test, attributes, target_classes, data_folder, is_train_set=False)
    X_train = X_train.drop(['recordingID'], axis=1)
    X_test = X_test.drop(['recordingID'], axis=1)
    y_train = y_train['mistake'].reset_index(drop=True)
    y_test = y_test['mistake'].reset_index(drop=True)
    # feature are selected ONLY in the training set
    selected_features = select_features(X_train, y_train, 0.05, data_folder)  # take only 5% of the best features n=~100

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    # print(selected_features)

    # scale data
    # Normalize/Scale only on train data, use the same scaler for test data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # model training
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        #params = grid_search(model, X_train, y_train)
        score = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        results.append(score)
        names.append(name)
        print('mean(score): ', np.mean(score), ', std(score):', np.std(score))
    plt.bar(names, results)
    plt.ylim(0, 1)
    plt.savefig('Confusion_matrix_and_ROC_Curve/models_comparing.png')
    plt.show()
    ROC_curve(models)
