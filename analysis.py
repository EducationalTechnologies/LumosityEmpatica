import data_helper
#import feature_extraction
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix



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

#create_diagram(list_of_participants) #Error if you load many zip-files...



list_of_attributes = ['BVP_std','GSR_mean','HRV_mean','IBI_mean','TMP_mean']

def regression_model(list_of_attributes):
    X = tabular_representation[list_of_attributes]
    y = tabular_representation['mistake']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0) #splitting the Data set into the Training Set and Test Set
    regressor = LogisticRegression()
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    score = regressor.score(X_test, y_test)

    #creating confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    plt.figure(0)
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.title('Accuracy Score: {0}'.format(score))
    plt.ylabel('Actual');
    plt.xlabel('Predicted');
    #plt.show()
    plt.savefig('confusion_matrix_and_ROC_Curve/confusion_matrix.png')
    print('Score:', score)

    #creating ROC Curve
    y_pred_proba = regressor.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.figure(1)
    plt.plot(fpr, tpr, label=" auc=" + str(auc))
    plt.text(0.3, 0.8, "ROC Curve")

    plt.plot([0, 1], [0, 1], color="black", linestyle="--")
    plt.text(0.6, 0.5, "Baseline")

    plt.title("Receiver Operating Characteristics")
    plt.ylabel('True positive rate');
    plt.xlabel('False positive rate');
    #plt.show()
    plt.savefig('confusion_matrix_and_ROC_Curve/ROC Curve.png')

regression_model(list_of_attributes)