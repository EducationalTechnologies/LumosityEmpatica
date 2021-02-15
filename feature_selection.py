import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE


def select_features(X, y, n_features, attributes, folder):
    string_attributes = '-'.join(attributes)
    file_features = f"{folder}/features/selected_features_" + string_attributes + "_" + str(n_features) + ".pkl"
    if os.path.exists(file_features):
        with open(file_features, "rb") as f:
            selected_features = pickle.load(f)
    else:
        estimator = DecisionTreeClassifier()
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)  # only take 0.05
        rfe.fit(X, y)

        # alternatively RFECV
        # rfe = RFECV(estimator=DecisionTreeClassifier(), step=0.01, scoring='accuracy', min_features_to_select=10)
        # rfe.fit(X, y)

        # return the selected features
        selected_features = X.columns[rfe.get_support()]

        # save selected features to avoid retraining again
        os.makedirs(f'{folder}/features', exist_ok=True)
        with open(f"{folder}/features/selected_features_" + string_attributes + "_" + str(n_features) + ".pkl",
                  "wb") as f:
            pickle.dump(selected_features, f)
    return selected_features
