import data_helper
import grid_search
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from feature_extraction import *
from feature_selection import select_features
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier


def get_models():
    models = dict()
    # models['SVC'] = SVC()
    models['KNC'] = KNeighborsClassifier()
    models['NB'] = GaussianNB()
    models['RFC'] = RandomForestClassifier()
    models['GBC'] = GradientBoostingClassifier()
    models['KNC'] = KNeighborsClassifier()
    # models['DTC'] = DecisionTreeClassifier()

    return models


def define_params(model):
    check_model = str(type(model))
    if ("SVC" in check_model):
        params = {'kernel': 'poly', 'gamma': 1, 'C': 10}
    elif ("KNeighborsClassifier" in check_model):
        params = {'leaf_size': 5}
    elif ("GaussianNB" in check_model):
        params = {'var_smoothing': 0.00000001}

    elif ("DecisionTreeClassifier" in check_model):
        params = {'criterion': 'gini'}
    elif ("RandomForestClassifier" in check_model):
        params = {'n_estimators': 300, 'max_features': 'auto', 'max_depth': 7, 'criterion': 'gini'}

    elif ("GradientBoostingClassifier" in check_model):
        params = {'subsample': 0.9,
                  'n_estimators': 100,
                  'min_samples_split': 0.3545454545454546,
                  'min_samples_leaf': 0.17272727272727273,
                  'max_features': 'log2',
                  'max_depth': 8,
                  'loss': 'deviance',
                  'learning_rate': 0.2,
                  'criterion': 'friedman_mse'}

    else:
        print("No 'params' defined for model")

    return params


def train_and_evaluate_model(model, params, X_train, y_train, X_test, y_test):
    for k, v in params.items():
        model.set_params(**{k: v})
    model_name = type(model).__name__
    # print(" ")
    # print("Training model {0}: ".format(model_name))
    model = Pipeline([('sampling', SMOTE(sampling_strategy='minority')), ('model', model)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score_acc = accuracy_score(y_test, y_pred)
    score_f1 = f1_score(y_test, y_pred)
    score_roc = roc_auc_score(y_test, y_pred)
    # plot_confusion_matrix(y_test, y_pred, score_f1, model_name)
    return score_acc, score_f1, score_roc


def plot_confusion_matrix(y_test, y_pred, score, model_name):
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues_r')
    score_f = '{0:.3g}'.format(score)
    plt.title('Confusion Matrix with {1}. F1-score: {0} '.format(score_f, model_name), fontsize=10)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(folder_plots + 'confusion_matrix_{0}.png'.format(model_name))
    plt.show()


def set_classifiers(model, models, params):
    for k, v in params.items():
        model.set_params(**{k: v})
    classifiers = list(models.values())

    return classifiers


# creating ROC curves for all models with using roc_auc_score
def ROC_curve(classifiers):
    table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
    for model in classifiers:
        model_name = type(model).__name__
        model.probability = True
        model = Pipeline([('sampling', SMOTE(sampling_strategy='minority')), ('model', model)])
        model = model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[::, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        table = table.append({'classifiers': model_name, 'fpr': fpr,
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
    plt.title('ROC Curves', fontweight='bold', fontsize=10)
    plt.legend(prop={'size': 9}, loc='lower right')
    plt.savefig(folder_plots + 'ROC_Curve.png', dpi=400)
    plt.show()


if __name__ == "__main__":
    data_folder = "manual_sessions/lumosity-dataset"
    ignore_files = ['ACC']
    to_exclude = ['OenName', 'RecordingID', 'ApplicationName']
    target_classes = ['mistake']
    tensor_data, annotations, attributes = data_helper.get_data_from_files(data_folder, ignore_files=ignore_files,
                                                                           res_rate=25,
                                                                           to_exclude=to_exclude)
    print("\nShape of the tensor_data is: " + str(np.shape(tensor_data)))
    print("Shape of the annotations is: " + str(np.shape(annotations)) + "\n")

    attributes = ['bvp', 'gsr', 'hrv', 'ibi', 'tmp']

    folder_plots = 'plots/'
    os.makedirs(folder_plots, exist_ok=True)

    X = tensor_data
    y = annotations.reset_index(drop=True)
    X = extract_df_with_features(X, y, attributes, target_classes, data_folder)
    # X = extract_basic_features(X, y, attributes)
    y_target = y['mistake']
    X_ids = X['recordingID']
    X = X.drop(['recordingID', 'mistake'], axis=1)

    # select the features with feature selection
    selected_features = select_features(X, y_target, 0.01, attributes, data_folder)
    for f in selected_features:
        if not f in X.columns.values:
            selected_features = selected_features.drop(f)
    X = X[selected_features]

    # add duration as a feature
    X.loc[:, 'duration'] = y.loc[:, 'duration']

    users_all = X_ids.unique()

    if len(users_all) > 1:
        scaler = MinMaxScaler()

        models = get_models()
        print('\nModel training ' + ', '.join(list(models.keys())))
        print('\nTesting with leave-one-user-out \n')

        # leave-one-out approach.
        dfResults = pd.DataFrame()
        for fold in users_all:
            print('- testing on fold: ' + fold)
            users_held = [fold]
            users_left = list(set(users_all) - set(users_held))
            y_train = y[y['recordingID'].isin(users_left)]['mistake']
            y_test = y[y['recordingID'].isin(users_held)]['mistake']
            X_train = X.loc[y_train.index.to_list()]
            X_test = X.loc[y_test.index.to_list()]

            # scale the features
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            resultsRow = {"fold": fold}
            for name, model in models.items():
                # params = grid_search.grid_search(model, X_train, y_train, X_test, y_test)
                params = define_params(model)
                score_acc, score_f1, score_roc = train_and_evaluate_model(model, params, X_train, y_train, X_test,
                                                                          y_test)
                result_tuple = (score_acc, score_f1, score_roc)
                classifiers = set_classifiers(model, models, params)
                resultsRow[name + '_acc'] = np.mean(score_acc)
                resultsRow[name + '_f1'] = np.mean(score_f1)
                resultsRow[name + '_roc-auc'] = np.mean(score_roc)
            dfResults = dfResults.append(resultsRow, ignore_index=True)

        print("\nSummary of the results:\n")
        # print(dfResults.mean().sort_values(ascending=False))
        print("\nMean Accuracy score: " + str(dfResults.loc[:, dfResults.columns.str.contains('acc')].mean().mean()))
        print("Mean F1 score: " + str(dfResults.loc[:, dfResults.columns.str.contains('f1')].mean().mean()))
        print("Mean ROC-AUC score: " + str(dfResults.loc[:, dfResults.columns.str.contains('roc')].mean().mean()))
    else:
        print("You need at least 2 sessions.")



    # Plot the principal components (PCA)
    dfX = X
    dfY = y_target.values
    dfX = StandardScaler().fit_transform(dfX)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(dfX)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, y_target], axis=1)
    #finalDf['duration'] = X['duration']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('principal component 1', fontsize=15)
    ax.set_ylabel('principal component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0.0, 1.0]
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['mistake'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

    # split the dataset between train and test
    # _train, y_train, X_test, y_test = data_helper.split_data_train_test(tensor_data, annotations,
    #                                                                     train_test_ratio=0.95, random_shuffling=False)
    # feature extraction both on the training set and the test set
    # X_train = extract_df_with_features(X_train, y_train, attributes, target_classes, data_folder)
    # X_test = extract_df_with_features(X_test, y_test, attributes, target_classes, data_folder)
    # X_train = X_train.drop(['recordingID'], axis=1)
    # X_test = X_test.drop(['recordingID'], axis=1)

    # y_train_target = y_train['mistake'].reset_index(drop=True)

    # feature are selected ONLY in the training set
    # selected_features = select_features(X_train, y_train_target, 0.25, attributes,
    #                                    data_folder)  # take only 5% of the best features n=~100
    # check if the selected attributes are in index otherwise drop them
    # for f in selected_features:
    #    if not f in X_train.columns.values or not f in X_test.columns.values:
    #        selected_features = selected_features.drop(f)
    # X_train = X_train[selected_features]
    # X_test = X_test[selected_features]
    # print(selected_features)

    # add feature duration of the annotation
    # X_train.loc[:, 'duration'] = y_train.loc[:, 'duration'].reset_index(drop=True)
    # X_test.loc[:, 'duration'] = y_test.loc[:, 'duration'].reset_index(drop=True)

    # set target and reset index
    # y_train = y_train['mistake'].reset_index(drop=True)
    # y_test = y_test['mistake'].reset_index(drop=True)

    # scale data
    # Normalize/Scale only on train data, use the same scaler for test data
    # scaler = MinMaxScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    # model training
    # models = get_models()
    # evaluate the models and store results
    # results, names = list(), list()

    # for name, model in models.items():
    #     # params = grid_search.grid_search(model, X_train, y_train, X_test, y_test)
    #     params = define_params(model)
    #     score_acc, score_f1 = train_and_evaluate_model(model, params, X_train, y_train, X_test, y_test)
    #     result_tuple = (score_acc, score_f1)
    #     results.append(result_tuple)
    #     classifiers = set_classifiers(model, models, params)
    #     names.append(name)
    #     print('mean(score_acc): ', np.mean(score_acc), ', std(score_acc):', np.std(score_acc))
    #     print('mean(score_f1): ', np.mean(score_f1), ', std(score_f1):', np.std(score_f1))
    #
    # for j in results:
    #     acc_score_list = []
    #     acc_score_list.append(j[0])
    # plt.xticks(fontsize=10)
    # plt.bar(names, acc_score_list)
    # plt.ylim(0, 1)
    # plt.title('Accuracy score', fontsize=10)
    # plt.savefig(folder_plots + 'acc_score_models_comparing.png')
    # plt.show()
    #
    # for j in results:
    #     f1_score_list = []
    #     f1_score_list.append(j[1])
    # plt.xticks(fontsize=10)
    # plt.bar(names, f1_score_list)
    # plt.ylim(0, 1)
    # plt.title('F-1 score', fontsize=10)
    # plt.savefig(folder_plots + 'f1_score_models_comparing.png')
    # plt.show()
    #
    # ROC_curve(classifiers)




