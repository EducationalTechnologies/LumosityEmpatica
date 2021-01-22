import data_helper
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
#print(tabular_representation['BVP_std'])

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
#print(tabular_representation)


#############################################################################################################
#Tanjas code changing

#setting graphs size
rc={'axes.labelsize': 10, 'font.size': 10, 'legend.fontsize': 10.0, 'axes.titlesize': 15 , "figure.figsize" : (8.27, 11.69)}
plt.rcParams.update(**rc)

np_list_of_participants = tabular_representation['recordingID'].unique()
list_of_participants = np_list_of_participants.tolist()
list_of_participants.append('all') #for creating summary graph

#creating graphs of pairslots for every participant and summary graph
def create_diagram(list_of_participants):
    for i in list_of_participants:
        participant = tabular_representation.loc[tabular_representation['recordingID'] == i]
        if (i != 'all'):
            s = sns.pairplot(participant, kind="reg", plot_kws={'line_kws': {'color': 'red'}})
            plt.suptitle('Participant ' + i, color='red', fontsize=25)
        else:
            s = sns.pairplot(tabular_representation, kind="reg", plot_kws={'line_kws': {'color': 'red'}})
            plt.suptitle('Summary graph', color='red', fontsize=25)
        fig = plt.gcf()  # get current figure
        fig.tight_layout()
        # plt.show()
        plt.savefig('plots/Participant_' + i + '.png', dpi=400)

create_diagram(list_of_participants)