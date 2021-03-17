import numpy as np
import pandas as pd
import data_helper
import os
import pomegranate as pg
from scipy.stats import norm
from feature_extraction import *
from feature_selection import select_features
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
from imblearn.over_sampling import SMOTE

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

if __name__ == "__main__":
    data_folder = "manual_sessions/lumosity-dataset"
    ignore_files = ['ACC']
    to_exclude = ['OenName', 'RecordingID', 'ApplicationName']
    target_class = 'mistake'
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


    X = extract_df_with_features(X, y, attributes, [target_class], data_folder)
    # X = extract_basic_features(X, y, attributes)
    y_target = y[target_class]
    X_ids = X['recordingID']
    X = X.drop(['recordingID', target_class], axis=1)

    # select the features with feature selection
    selected_features = select_features(X, y_target, 0.01, attributes, data_folder)     # doesn't work with 10% = (0.1)
    for f in selected_features:
        if not f in X.columns.values:
            selected_features = selected_features.drop(f)
    X = X[selected_features]
    print("Number of selected features:", len(selected_features))


    # add duration as a feature
    #X['duration'] = y['duration']
    # X = X[['duration']]  # only duration as feature

    y.loc[:, 'score'] = (1 - y.loc[:, target_class]) / y.loc[:, 'duration']
    score_values_nozeros = y[y.score > 0].score.values
    mu, std = norm.fit(score_values_nozeros)
    score_values = y.score.values
    y.loc[:, 'score_normalized'] = gaussian(score_values, mu, std)
    y.loc[:, 'score_norm_binary'] = pd.cut(y.loc[:, 'score_normalized'], bins=2, labels=[0,1])
    # override target class
    target_class = 'score_norm_binary'
    target = y[target_class].astype('int32').values

    sessions_all = X_ids.unique()

    dfResults = pd.DataFrame()
    if len(sessions_all) > 1:
        scaler = MinMaxScaler()
        print('\nTesting with leave-one-session-out \n')
        # leave-one-out approach.
        dfResults = pd.DataFrame()
        for fold in sessions_all:
            print('- testing on fold: ' + fold)
            users_held = [fold]
            users_left = list(set(sessions_all) - set(users_held))
            y_train = y[y['recordingID'].isin(users_left)][target_class]
            y_test = y[y['recordingID'].isin(users_held)][target_class]
            X_train = X.loc[y_train.index.to_list()]
            X_test = X.loc[y_test.index.to_list()]
            # scale the features
            if X_train.ndim < 2:
                X_train = X_train.values.reshape(-1, 1)
                X_test = X_test.values.reshape(-1, 1)
            else:
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                # print(Counter(y_train))
                sampler = SMOTE(sampling_strategy='minority')
                X_train, y_train = sampler.fit_sample(X_train, y_train)
                # print(Counter(y_train))

            resultsRow = {"fold": fold}
            X_train = np.expand_dims(X_train, axis=0)
            labels = np.expand_dims(y_train, axis=0)
            n_components = len(np.unique(labels))

            model = pg.HiddenMarkovModel.from_samples(pg.MultivariateGaussianDistribution,
                                                      n_components=n_components,
                                                      X=X_train,
                                                      labels=labels,
                                                      algorithm='labeled')

            y_pred = model.predict(X_test)
            resultsRow['HMM_acc'] = accuracy_score(y_test, y_pred)
            # accuracy_score(y_test, y_pred) is the same as model.score(X_test, y_test))
            resultsRow['HMM_f1'] = f1_score(y_test, y_pred)
            resultsRow['HMM_roc-auc'] = roc_auc_score(y_test, y_pred)
            dfResults = dfResults.append(resultsRow, ignore_index=True)

        print("\nSummary of the results:")
        mean_acc = '{0:.3g}'.format(dfResults.loc[:, dfResults.columns.str.contains('acc')].mean().mean())
        mean_f1 = '{0:.3g}'.format(dfResults.loc[:, dfResults.columns.str.contains('f1')].mean().mean())
        mean_roc = '{0:.3g}'.format(dfResults.loc[:, dfResults.columns.str.contains('roc')].mean().mean())
        print("\nMean Accuracy score: " + mean_acc)
        print("Mean F1 score: " + mean_f1)
        print("Mean ROC-AUC score: " + mean_roc)























