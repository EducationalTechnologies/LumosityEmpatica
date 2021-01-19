import data_helper
import numpy as np

train_folder = "manual_sessions/lumosity-dataset"
ignore_files = []
to_exclude = ['ACC','OenName','RecordingID', 'ApplicationName'] #exclude the following attributes in generating the Tensor
target_classes = ['mistake']
tensor_data, annotations, attributes = data_helper.get_data_from_files(train_folder, ignore_files=ignore_files, res_rate=25,
                                                           to_exclude=to_exclude)

print("Shape of the tensor_data is: " + str(np.shape(tensor_data)))
print("Shape of the annotation is: " + str(np.shape(annotations)))

attributes = ['BVP', 'GSR', 'HRV', 'IBI', 'TMP']

tabular_representation = annotations[['recordingID', 'mistake', 'end', 'start', 'duration']].copy()

# BVP
# matrix BVP tensor_data[:,:,0]
tabular_representation['BVP_std'] = np.std(tensor_data[:,:,0],axis=1)

# GSP
# matrix GSR tensor_data[:,:,1]
tabular_representation['GSR_mean'] = np.mean(tensor_data[:,:,1],axis=1)

# HRV
# matrix HRV tensor_data[:,:,2]
tabular_representation['HRV_mean'] = np.mean(tensor_data[:,:,2],axis=1)

# IBI
# matrix IBI tensor_data[:,:,3]
tabular_representation['IBI_mean'] = np.mean(tensor_data[:,:,3],axis=1)

# TMP
# matrix TMP tensor_data[:,:,4]
tabular_representation['TMP_mean'] = np.mean(tensor_data[:,:,4],axis=1)

# Temperature column
print(tabular_representation)
