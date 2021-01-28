import data_helper
import numpy as np
import pandas as pd
from tsfresh import *


def extract_df_with_features(tensor, annotations, attributes, target_classes):
    m, n, r = tensor.shape
    out_arr = np.column_stack((np.repeat(np.arange(m), n), tensor.reshape(m * n, -1)))
    attributes = ['interval'] + attributes
    out_df = pd.DataFrame(out_arr, columns=attributes)
    out_df['time'] = out_df.groupby(['interval']).cumcount()
    extracted_features = extract_features(out_df, column_id="interval", column_sort="time", column_kind=None,
                                          column_value=None)
    extracted_features[target_classes] = annotations[target_classes]
    extracted_features.index = extracted_features.index.astype('int64')
    # drop the features where all NaNs
    extracted_features = extracted_features.dropna(axis=1, how='all')
    # fill the NaNs in the remaining features
    extracted_features = extracted_features.fillna(method='bfill').fillna(method='ffill')
    extracted_features.loc[:,'recordingID'] = annotations['recordingID']

    # select relevant features
    #cor = extracted_features.corr()
    #cor_target = abs(cor[target_classes[0]]) # this assumes there is only one target
    #relevant_features = cor_target.sort_values(ascending=False)[:no_of_features].index.to_list()
    #filtered_features = extracted_features[relevant_features].fillna(method='bfill').fillna(method='ffill')

    #print("The selected N="+str(no_of_features)+" features are ")
    #print(filtered_features.columns)


    return extracted_features

