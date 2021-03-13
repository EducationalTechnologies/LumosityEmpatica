import numpy as np
import pandas as pd
import data_helper
import os
import pomegranate as pg
from scipy.stats import norm
from feature_extraction import *
from feature_selection import select_features
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    print("Duration was added into features")
    X['duration'] = y['duration']
    # X = X[['duration']]                                               # only duration as feature

    y.loc[:, 'score'] = (1 - y.loc[:, target_class]) / y.loc[:, 'duration']
    score_values_nozeros = y[y.score > 0].score.values
    mu, std = norm.fit(score_values_nozeros)
    score_values = y.score.values
    y.loc[:, 'score_normalized'] = gaussian(score_values, mu, std)
    y.loc[:, 'score_norm_binary'] = pd.cut(y.loc[:, 'score_normalized'], bins=2, labels=[0, 1]).astype('int32')
    # override target class
    target_class = 'score_norm_binary'
    target = y[target_class].astype('int32').values
    y.score.hist(bins=50)
    plt.ylabel('Frequency')
    plt.xlabel('Score = (1-mistake)/duration')
    #plt.show()


    users_all = X_ids.unique()
    if len(users_all) > 1:
        scaler = MinMaxScaler()
        #print('\nTesting with leave-one-session-out \n')
        # leave-one-out approach.
        dfResults = pd.DataFrame()
        for fold in users_all:
            #print('- testing on fold: ' + fold)
            users_held = [fold]
            users_left = list(set(users_all) - set(users_held))
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



    X_train = np.expand_dims(X_train, axis=0)
    labels = np.expand_dims(y_train, axis=0)

    print("Shape of X_train:", X_train.shape)
    print("Shape of labels:", labels.shape)


    model = pg.HiddenMarkovModel.from_samples(pg.MultivariateGaussianDistribution,
                                              n_components=len(np.unique(labels)),
                                              X=X_train,
                                              labels=labels,
                                              algorithm = 'labeled')

    #X_test = np.expand_dims(X_test, axis=0)
    #y_test = np.expand_dims(y_test, axis=0)

    print(" ")
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)

    # print("viterbi:",model.viterbi(X_test))
    # print("maximum_a_posteriori:", model.maximum_a_posteriori(X_test))
    # print("predict_proba:", model.predict_proba(X_test))
    print("Dense transition matrix:")
    print(model.dense_transition_matrix())
    print("Emission and transition matrix:")
    print(model.forward_backward(X_test))
    print("Log probability:", model.log_probability(X_test))
    print("Number of states:", model.state_count())                 # Start, 1, 0, End
    y_pred = model.predict(X_test)
    print("Score:", model.score(X_test, y_test))
    # print("y_pred:", y_pred)
    # print("y_test:", y_test.values)



    # parameters are not like in analysis.py: 'score = model.score(y_pred, y_test))'
    # score() Return the accuracy of the model on a data set.
    # Parameters:
    # X[numpy.ndarray, shape=(n, d)] The values of the data set
    # y[numpy.ndarray, shape=(n,)] The labels of each value























