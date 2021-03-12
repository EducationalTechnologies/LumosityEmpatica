import numpy as np
import pandas as pd
import data_helper
import os
import pomegranate as pg
from scipy.stats import norm

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

    df = y.copy()

    df['bvp'] = np.mean(tensor_data[:, :, 0], axis=1)
    df['gsr'] = np.mean(tensor_data[:, :, 1], axis=1)
    df['hrv'] = np.mean(tensor_data[:, :, 2], axis=1)
    df['ibi'] = np.mean(tensor_data[:, :, 3], axis=1)
    df['tmp'] = np.mean(tensor_data[:, :, 4], axis=1)

    observations = df[attributes].values

    y.loc[:, 'score'] = (1 - y.loc[:, target_class]) / y.loc[:, 'duration']
    score_values_nozeros = y[y.score > 0].score.values
    mu, std = norm.fit(score_values_nozeros)
    score_values = y.score.values
    y.loc[:, 'score_normalized'] = gaussian(score_values, mu, std)
    y.loc[:, 'score_norm_binary'] = pd.cut(y.loc[:, 'score_normalized'], bins=2, labels=[0, 1]).astype('int32')
    # override target class
    target_class = 'score_norm_binary'
    target = y[target_class].astype('int32').values

    X = np.expand_dims(observations, axis=0)
    labels = np.expand_dims(target, axis=0)

    print("Shape of X:", X.shape)
    print("Shape of labels:", labels.shape)


    model = pg.HiddenMarkovModel.from_samples(pg.MultivariateGaussianDistribution,
                                            n_components=len(np.unique(labels)),
                                            X=X,
                                            labels=labels,
                                            algorithm='labeled')

    test = np.array([[[0.1370942, 65.08173, 0.9219177, 30.369999999999994],
                    [0.14720542857142857, 65.08173, 0.9219177, 30.369999999999994],
                    [0.1370942, 76.64283057142856, 0.7942332199999999, 30.369999999999994]]])
    test_y = np.array([[1,0,0]])

    # print("viterbi:",model.viterbi(test))
    # print("maximum_a_posteriori:", model.maximum_a_posteriori(test))
    # print("predict_proba:", model.predict_proba(test))
    print("Dense transition matrix:", model.dense_transition_matrix())
    print("Emission and transition matrix:", model.forward_backward(test))
    print("Number of states:", model.state_count())
    print("Log probability:", model.log_probability(test))
    print("Predicted test:", model.predict(test))
    print("Score:", model.score(test, test_y))




    # uniqueId = df["recordingID"].unique()
    # observations = []
    # mistakes = []
    # mistakes_partcp = []
    # for id_ in uniqueId:
    #     df_part = df.loc[df['recordingID'] == id_]
    #     mistake = df_part['mistake']
    #     mistake_list = []
    #     participant = [df_part['gsr'], df_part['bvp'], df_part['ibi'], df_part['tmp']]
    #     values = []
    #     for column in participant:
    #         columns = []
    #         for value in column:
    #             columns.append(value)
    #         values.append(columns)
    #     observations.append(values)
    #     for value in mistake:
    #         mistakes.append(value)                  # mistakes = [mistakes_1_to_1927]
    #
    # end_list = []
    # for partc in observations:
    #     for a in range(len(partc[0])):
    #         temp = []
    #         for b in range(len(observations[0])):
    #             temp.append(partc[b][a])
    #         end_list.append(temp)
    # observations = end_list                 #[[GSR_1,BVP_1,IBI_1,TMP_1],...[GSR_1997,BVP_1927,IBI_1991,TMP_1927]]



    #X = np.array([np.array(x) for x in observations], dtype=object)
    #labels = np.array(mistakes)

    #X = np.expand_dims(X, axis=0)






















