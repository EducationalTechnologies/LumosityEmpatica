import numpy as np
import data_helper
import os
from hmmlearn import hmm

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

    # Let's consider only GSR
    df = annotations[['recordingID', 'mistake', 'end', 'start', 'duration']].copy()
    df['gsr'] = np.mean(tensor_data[:, :, 1], axis=1)

    # Initializing 
    # X = np.concatenate([X1, X2])
    # lengths = [len(X1), len(X2)]
    # hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    # hmm_model.fit(X, lengths)

    #y_pred = remodel.predict(X)