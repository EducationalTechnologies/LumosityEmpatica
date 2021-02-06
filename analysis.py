import data_helper
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from feature_extraction import extract_df_with_features
from feature_selection import select_features
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV


# from sklearn.model_selection import RepeatedStratifiedKFold, KFold
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import cross_val_score

def get_models():
    models = dict()
    models['support_vector_machines'] = SVC()
    models['random_forest'] = RandomForestClassifier()
    models['gradient_boosting'] = GradientBoostingClassifier()
    return models


def train_and_evaluate_model(model, params, X_train, y_train, X_test, y_test):
    for k, v in params.items():
        model.set_params(**{k: v})
    print("Training model " + str(model))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = accuracy_score(y_test, y_pred)
    return scores


def grid_search(model, X, y):
    check_model = str(type(model))

    if ("SVC" in check_model):
        print(" ")
        print("Analysing of best parameters for", model, "...")
        param_grid = {'C': [0.1, 1, 10, 100],
                      'gamma': [1, 0.1, 0.01, 0.001],
                      'kernel': ['rbf', 'poly', 'sigmoid']}
        # param_grid = {'C': [0.1, 1]}  # for fast code testing

    elif ("RandomForestClassifier" in check_model):
        print(" ")
        print("Analysing of best parameters for", model, "...")
        param_grid = {'n_estimators': [10, 100, 120],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'max_depth': [4, 5, 6, 7],
                      'criterion': ['gini', 'entropy']}
        param_grid = {'n_estimators': [10, ]}  # for fast code testing

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

        param_grid = {'n_estimators': [10, 100]}  # for fast code testing
    else:
        print("Define grid_param in elif-loop for new model, which you added to models-set")

    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=1)
    grid_results = grid.fit(X, y)  # X_train, y_train ???
    print(
        "Best score for {2}: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_, model))
    params = grid_results.best_params_
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    return params


# creating ROC curves for all models with using roc_auc_score
def ROC_curve(X, y):
    # classifiers = list(models.values()) #error by using models with different n_estimators
    classifiers = [SVC(kernel='rbf'), RandomForestClassifier(n_estimators=10), GradientBoostingClassifier()]
    table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
    for cls in classifiers:
        cls.probability = True
        model = cls.fit(X, y)
        yproba = model.predict_proba(X_test)[::, 1]
        fpr, tpr, _ = roc_curve(y_test, yproba)
        auc = roc_auc_score(y_test, yproba)
        table = table.append({'classifiers': cls.__class__.__name__, 'fpr': fpr,
                              'tpr': tpr, 'auc': auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    table.set_index('classifiers', inplace=True)
    for i in table.index:
        plt.plot(table.loc[i]['fpr'],
                 table.loc[i]['tpr'], label="{}, AUC={:.3f}".format(i, table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.text(0.6, 0.5, "Baseline")
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title('ROC Curves', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 9}, loc='lower right')
    plt.savefig('confusion_matrix_and_ROC_Curve/ROC_Curve' + i + '.png', dpi=400)
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
        params = grid_search(model, X_train, y_train)
        score = train_and_evaluate_model(model, params, X_train, y_train, X_test, y_test)
        results.append(score)
        names.append(name)
        print(name, '- mean(score): ', np.mean(score), ', std(score):', np.std(score))

    # plot models performance for comparison
    # plt.boxplot(results, labels=names, showmeans=True)  # not working
    plt.bar(names, results)
    plt.show()
    # ROC_curve()   # not working

    # # setting graphs size
    # rc = {'axes.labelsize': 10, 'font.size': 10, 'legend.fontsize': 10.0, 'axes.titlesize': 15,
    #       "figure.figsize": (8.27, 11.69)}
    # plt.rcParams.update(**rc)
    #
    # np_list_of_participants = tabular_representation['recordingID'].unique()
    # list_of_participants = np_list_of_participants.tolist()
    #
    # list_of_participants.append('all')  # for creating summary graph
    #
    #
    # # creating graphs of pairslots for every participant and summary graph
    # def create_diagram(list_of_participants):
    #     for i in list_of_participants:
    #         participant = tabular_representation.loc[tabular_representation['recordingID'] == i]
    #         if (i != 'all'):
    #             s = sns.pairplot(participant, kind="reg", plot_kws={'line_kws': {'color': 'red'}})
    #             plt.suptitle('Participant ' + i, color='red', fontsize=25)
    #         else:
    #             s = sns.pairplot(tabular_representation, kind="reg", plot_kws={'line_kws': {'color': 'red'}})
    #             plt.suptitle('Summary graph', color='red', fontsize=25)
    #         fig = plt.gcf()  # get current figure
    #         fig.tight_layout()
    #         # plt.show()
    #         plt.savefig('plots/Participant_' + i + '.png', dpi=400)
    #
    #
    # # create_diagram(list_of_participants)
    # # list_of_attributes = tabular_representation.columns[~tabular_representation.columns.isin(target_classes + ['recordingID'])][:10]
    #
    # ##############################################################################################################
    # # New changes for classification models
    # """
    #     # @todo scale the features
    #     # Normalize/Scale only on train data. Use that scaler to later scale valid and test data
    #     scaler = MinMaxScaler()
    #     X = scaler.fit(X)
    #     """
    #

    #
    # # #get test data
    # # test_folder = "manual_sessions/lumosity-dataset/test"
    # # ignore_files = []
    # # to_exclude = ['ACC', 'OenName', 'RecordingID','ApplicationName']
    # # target_classes = ['mistake']
    # # tensor_data, annotations, attributes = data_helper.get_data_from_files(test_folder, ignore_files=ignore_files,
    # #                                                                        res_rate=25,
    # #                                                                        to_exclude=to_exclude)
    # # print("Shape of the tensor_data is: " + str(np.shape(tensor_data)))
    # # print("Shape of the annotation is: " + str(np.shape(annotations)))
    # # attributes = ['bvp', 'gsr', 'hrv', 'ibi', 'tmp']
    #
    # # tabular_representation_test = extract_df_with_features(tensor_data, annotations, attributes, target_classes,
    # #                                                       test_folder)
    #
    # # X_test = tabular_representation_test.drop(['mistake', 'recordingID'], axis=1)
    # # y_test = tabular_representation_test['mistake']
    # # X_test = X_test[selected_features]
    #
    # def get_models():
    #     models = dict()
    #     models['SVC_rbf'] = SVC()
    #     models['random_forest'] = RandomForestClassifier()
    #     models['gradient_boosting'] = GradientBoostingClassifier()
    #     return models
    #
    #
    # ''' #VERSION 1
    #     #Here below is a version with splitting X from train-folder into 2 subsets.
    #     #When and how should we use X_test and y_test, which we extracted from test-folder, when we already trained our model
    #     #and got the score with some test (validation) set?
    #     #Here I splitted X manually to get access only to train data, which should be scaled, so I didnt use cross-validation.
    # '''
    #
    #
    # def evaluate_model(model, X, y, X_test, y_test):
    #     # scale on train data
    #     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=1)
    #     scaler = MinMaxScaler()
    #     scaler.fit(X_train)
    #     X_train = scaler.transform(X_train)
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_val)
    #     scores = accuracy_score(y_val, y_pred)
    #     return scores
    #
    #
    # ''' VERSION 2
    #     #Here below I used our data from test and train folders and calculated the score like I did
    #     #it in logistic regression before. Here you could also get the confusion matrix for each model.
    #     #Is it the way we should use out test-set?
    #
    # def evaluate_model(model, X, y, X_test, y_test):
    #
    #     # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #     # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    #
    #     # scale on train data
    #     scaler = MinMaxScaler()
    #     scaler.fit(X)
    #     X = scaler.transform(X)
    #     model.fit(X, y)
    #     y_pred = model.predict(X_test)
    #     scores = accuracy_score(y_test, y_pred)
    #
    #
    #     # confusion matrix, uncomment if you want to see it
    #     cm = confusion_matrix(y_test, y_pred)
    #     sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    #     plt.title('Accuracy Score with {1} model: {0} '.format(scores, model), fontsize=10)
    #     plt.ylabel('Actual')
    #     plt.xlabel('Predicted')
    #     plt.show()
    #     plt.savefig('confusion_matrix_and_ROC_Curve/confusion_matrix.png')
    #     return scores
    #
    # '''
    #
    # '''
    #     Do we need this function below in evaluate_model?
    #     We could run it once, select the best parameters for models and change models in get_models()
    #     But look at grid_results = grid.fit(X, y) Should be here X_train, y_train instead of X,y?
    # '''
    #
    #
