import numpy as np
import pandas as pd
from tsfresh import *
import pickle
import os


def extract_df_with_features(tensor, annotations, attributes, target_classes, folder):
    folder_features = f"{folder}/features/"
    # string_attributes = '-'.join(annotations['recordingID'].unique())
    file_features = f"{folder}/features/extracted_features_N"+str(len(annotations))+".pkl"
    if os.path.exists(file_features):
        with open(file_features, "rb") as f:
            extracted_features = pickle.load(f)
    else:
        m, n, r = tensor.shape
        out_arr = np.column_stack((np.repeat(np.arange(m), n), tensor.reshape(m * n, -1)))
        attributes = ['interval'] + attributes
        out_df = pd.DataFrame(out_arr, columns=attributes)
        out_df['time'] = out_df.groupby(['interval']).cumcount()
        ef_df = extract_features(out_df, column_id="interval", column_sort="time", column_kind=None,
                                 column_value=None)
        ef_df[target_classes] = annotations[target_classes]
        ef_df.index = ef_df.index.astype('int64')
        ef_df = ef_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        # drop the features where all NaNs
        extracted_features = ef_df.dropna(axis=1, how='all')
        # fill the NaNs in the remaining features
        extracted_features = extracted_features.fillna(method='bfill').fillna(method='ffill')
        extracted_features.loc[:, 'recordingID'] = annotations['recordingID']
        os.makedirs(folder_features, exist_ok=True)
        with open(file_features, "wb") as f:
            pickle.dump(extracted_features, f)

    return extracted_features


def extract_basic_features(tensor, annotations, attributes):
    dict_functions = {'_mean' : np.mean}
    df = pd.DataFrame()
    for f in dict_functions:
        attrs = list(map(lambda x: x + f, attributes))
        values = dict_functions[f](tensor, axis=1)
        tmp = pd.DataFrame(values, columns=attrs)
        df = pd.concat([df, tmp], axis=1)
    df.loc[:, 'recordingID'] = annotations['recordingID']
    df.loc[:, 'mistake'] = annotations['mistake']
    return df
