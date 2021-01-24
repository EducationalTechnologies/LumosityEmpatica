import data_helper
import numpy as np
import pandas as pd
from tsfresh import *


train_folder = "manual_sessions/lumosity-dataset"
ignore_files = []
to_exclude = ['ACC','OenName','RecordingID', 'ApplicationName'] #exclude the following attributes in generating the Tensor
target_class = 'mistake'
tensor_data, annotations, attributes = data_helper.get_data_from_files(train_folder, ignore_files=ignore_files, res_rate=25,
                                                           to_exclude=to_exclude)

print("Shape of the tensor_data is: " + str(np.shape(tensor_data)))
print("Shape of the annotation is: " + str(np.shape(annotations)))

annotations = annotations.reset_index(drop=True).reset_index()

#attributes = ['BVP', 'GSR', 'HRV', 'IBI', 'TMP']
attributes = ['interval','bvp', 'gsr', 'hrv', 'ibi', 'tmp']

if __name__=="__main__":
    m,n,r = tensor_data.shape
    out_arr = np.column_stack((np.repeat(np.arange(m),n),tensor_data.reshape(m*n,-1)))
    out_df = pd.DataFrame(out_arr,columns=attributes)
    out_df['time'] = out_df.groupby(['interval']).cumcount()
    extracted_features = extract_features(out_df,column_id="interval", column_sort="time", column_kind=None, column_value=None)

    # Correlation with output variable
    no_of_features = 100
    extracted_features[target_class] = annotations[target_class]
    cor = extracted_features.corr()
    cor_target = abs(cor[target_class])
    relevant_features = cor_target.sort_values(ascending=False)[:no_of_features].index.to_list()
    filtered_features = extracted_features[relevant_features]