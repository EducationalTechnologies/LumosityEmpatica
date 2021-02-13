import data_helper
import grid_search
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from feature_extraction import extract_df_with_features
from feature_selection import select_features
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def get_models():
    models = dict()
    models['SVC'] = SVC()
    models['RFC'] = RandomForestClassifier()
    models['GBC'] = GradientBoostingClassifier()
    '''
    models['KNC'] = KNeighborsClassifier()
    models['DTC'] = DecisionTreeClassifier()
    models['NB'] = GaussianNB()
    '''
    return models

def define_params(model):
    check_model = str(type(model))
    if ("SVC" in check_model):
        params = {'kernel': 'poly', 'gamma': 1, 'C': 10}

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
        '''
    elif ("KNeighborsClassifier" in check_model):
        params = {'leaf_size' : 5}

    elif ("DecisionTreeClassifier" in check_model):
        params = {'criterion': 'gini'}

    elif ("GaussianNB" in check_model):
        params = {'var_smoothing': 0.00000001}
        '''
    else:
        print("No 'params' defined for model")

    return params



def train_and_evaluate_model(model, params,  X_train, y_train, X_test, y_test):
    for k, v in params.items():
      model.set_params(**{k: v})
    model_name = type(model).__name__
    print(" ")
    print("Training model {0}: ".format(model_name))
    model = Pipeline([('sampling', SMOTE(sampling_strategy='minority')), ('model', model)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score_acc = accuracy_score(y_test, y_pred)
    score_f1 = f1_score(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, score_f1, model_name)
    return score_acc, score_f1


def plot_confusion_matrix(y_test, y_pred, score, model_name):
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.title('F1_score with {1} model: {0} '.format(score, model_name), fontsize=10)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(folder_plots + 'confusion_matrix_{0}.png'.format(model_name))
    plt.show()



def set_classifiers(model, params):
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
    ignore_files = []
    to_exclude = ['ACC', 'OenName', 'RecordingID', 'ApplicationName']
    target_classes = ['mistake']
    tensor_data, annotations, attributes = data_helper.get_data_from_files(data_folder, ignore_files=ignore_files,
                                                                           res_rate=25,
                                                                           to_exclude=to_exclude)
    print("Shape of the tensor_data is: " + str(np.shape(tensor_data)))
    print("Shape of the annotation is: " + str(np.shape(annotations)) + "\n")

    attributes = ['bvp', 'gsr', 'hrv', 'ibi', 'tmp']

    folder_plots = 'plots/'
    os.makedirs(folder_plots, exist_ok=True)

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
    results, names= list(), list()

    for name, model in models.items():
        #params = grid_search.grid_search(model, X_train, y_train, X_test, y_test)
        params = define_params(model)
        score_acc, score_f1 = train_and_evaluate_model(model, params, X_train, y_train, X_test, y_test)
        result_tuple = (score_acc, score_f1)
        results.append(result_tuple)
        classifiers = set_classifiers(model, params)
        names.append(name)
        print('mean(score_acc): ', np.mean(score_acc), ', std(score_acc):', np.std(score_acc))
        print('mean(score_f1): ', np.mean(score_f1), ', std(score_f1):', np.std(score_f1))

    for j in results:
        acc_score_list = []
        acc_score_list.append(j[0])
    plt.xticks(fontsize=10)
    plt.bar(names, acc_score_list)
    plt.ylim(0, 1)
    plt.title('Accuracy score', fontsize=10)
    plt.savefig(folder_plots + 'acc_score_models_comparing.png')
    plt.show()

    for j in results:
        f1_score_list = []
        f1_score_list.append(j[1])
    plt.xticks(fontsize=10)
    plt.bar(names, f1_score_list)
    plt.ylim(0, 1)
    plt.title('F-1 score', fontsize=10)
    plt.savefig(folder_plots + 'f1_score_models_comparing.png')
    plt.show()

    ROC_curve(classifiers)

