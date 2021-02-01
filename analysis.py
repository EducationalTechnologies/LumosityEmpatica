import data_helper
import feature_extraction
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
from sklearn.metrics import confusion_matrix
from feature_extraction import extract_df_with_features
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


if __name__ == "__main__":
    train_folder = "manual_sessions/lumosity-dataset"
    ignore_files = []
    # exclude the following attributes in generating the Tensor
    to_exclude = ['ACC', 'OenName', 'RecordingID',
                  'ApplicationName']
    target_classes = ['mistake']
    tensor_data, annotations, attributes = data_helper.get_data_from_files(train_folder, ignore_files=ignore_files,
                                                                           res_rate=25,
                                                                           to_exclude=to_exclude)

    print("Shape of the tensor_data is: " + str(np.shape(tensor_data)))
    print("Shape of the annotation is: " + str(np.shape(annotations)))
    attributes = ['bvp', 'gsr', 'hrv', 'ibi', 'tmp']
    tabular_representation = extract_df_with_features(tensor_data, annotations, attributes, target_classes)
    #for col in tabular_representation.columns:
        #print(col)

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


    #list_of_attributes = ['BVP_std','GSR_mean','HRV_mean','IBI_mean','TMP_mean']
    list_of_attributes = tabular_representation.columns[~tabular_representation.columns.isin(target_classes + ['recordingID'])][:20]

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
        # plt.show()
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
        # plt.show()
        plt.savefig('confusion_matrix_and_ROC_Curve/ROC_Curve.png')

    #regression_model(list_of_attributes)

################################################################################################################
#New changes for regression models

    #tabular_representation = tabular_representation.iloc[290:] #for quick code testing!
    #tabular_representation = tabular_representation.drop(tabular_representation.iloc[:, 0:2180], axis=1) #for quick code testing

    X = tabular_representation.drop(['mistake', 'recordingID'], axis=1)
    y = tabular_representation['mistake']

    def get_models():
        models = dict()
        '''
        # logistic regression
        rfe = RFECV(estimator=LogisticRegression())
        model = DecisionTreeClassifier()
        models['logistic regression'] = Pipeline(steps=[('s', rfe), ('m', model)])

        # perceptron
        rfe = RFECV(estimator=Perceptron())
        model = DecisionTreeClassifier()
        models['perceptron'] = Pipeline(steps=[('s', rfe), ('m', model)])

        # cart
        rfe = RFECV(estimator=DecisionTreeClassifier())
        model = DecisionTreeClassifier()
        models['cart'] = Pipeline(steps=[('s', rfe), ('m', model)])
        
        # rf
        rfe = RFECV(estimator=RandomForestClassifier())
        model = DecisionTreeClassifier()
        models['random forest'] = Pipeline(steps=[('s', rfe), ('m', model)])

        # gbm
        #rfe = RFECV(estimator=GradientBoostingClassifier())
        #model = DecisionTreeClassifier()
        #models['gbm'] = Pipeline(steps=[('s', rfe), ('m', model)])
        '''
        #svc linear
        rfe = RFECV(estimator=SVC(kernel='linear'), step=1, cv=3, scoring='f1')
        model = SVC(kernel='linear')
        models['svm linear'] = Pipeline(steps=[('s', rfe), ('m', model)])

        return models

    # evaluate a give model using cross-validation
    def evaluate_model(model, X, y):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        return scores

    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print(name, '- mean(score): ',np.mean(scores),', std(score):', np.std(scores))
    # plot model performance for comparison
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()
