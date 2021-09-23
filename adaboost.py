from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np

import pickle


def read_data(train_data, test_data):
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)
    return train_data, test_data


# select the features and target
def select_features(train_data):
    # select features
    X = train_data.drop("next_month_plan", axis=1).copy()
    y = train_data["next_month_plan"].copy()
    return X, y


# convert categorical columns to numeric
def encode_categorical_columns(train_data):
    # categorical columns
    categorical_columns = [
        "device_type",
        "device_category",
        "gender",
        "district_name",
    ]

    X_encode = pd.get_dummies(train_data, columns=categorical_columns)
    return X_encode


# Split the data into training and testing sets
def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


# find adaboost parameters using grid search
def adaboost_parameter_tuning(X_train, y_train, X_test, y_test):
    # adaboost classifier
    adaboost_clf = AdaBoostClassifier(random_state=42)
    # parameters
    parameters = {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    # grid search
    grid_search = GridSearchCV(
        adaboost_clf, param_grid=parameters, cv=5, scoring="f1_macro"
    )
    grid_search.fit(X_train, y_train)
    # save the model
    pickle.dump(grid_search, open("model/grid_search_model.sav", "wb"))
    # predict
    predictions = grid_search.predict(X_test)
    # calculate the f1 score
    f1_score_macro = calculate_f1_score(y_test, predictions)
    print("f1 score is: ", f1_score_macro)
    print("confusion matrix: ")
    print(confusion_matrix(y_test, predictions))
    # return the model and predictions
    return grid_search, predictions


# calculate the macro f1 score
def calculate_f1_score(y_test, predictions):
    f1_score_macro = f1_score(y_test, predictions, average="macro")
    return f1_score_macro


def main():
    # read data
    train_data, test_data = read_data(
        "input/train_dataset.csv", "input/test_dataset.csv"
    )
    # map target to categorical
    map_dict_target = {
        "PKG1": 1,
        "PKG2": 2,
        "PKG3": 3,
        "PKG4": 4,
        "PKG5": 5,
        "PKG6": 6,
        "PKG7": 7,
        "PKG8": 8,
    }
    train_data["next_month_plan"].replace(map_dict_target, inplace=True)

    print("target value distribution in the Train data: ")
    print(train_data["next_month_plan"].value_counts())

    # delete age_group column
    rm_column = [
        "age_group",
        "vusage_onnet_avg",
        "vusage_offnet_avg",
        "add_on_tot_rental",
        "add_on_count",
    ]
    train_data.drop(rm_column, axis=1, inplace=True)
    test_data.drop(rm_column, axis=1, inplace=True)

    # remove all rows with null values
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    X, y = select_features(train_data)
    # encode categorical columns
    X_encode = encode_categorical_columns(X)
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X_encode, y, test_size=0.2, random_state=42
    )
    # parameters
    # adaboost classifier
    grid_search, predictions = adaboost_parameter_tuning(
        X_train, y_train, X_test, y_test
    )

    # test data
    test_data = encode_categorical_columns(test_data)
    test_data = test_data[X_train.columns]
    test_predictions = grid_search.predict(test_data)

    submission = pd.DataFrame(
        {
            "primary_identifier": test_data["primary_identifier"],
            "next_month_plan": test_predictions,
        }
    )
    submission.to_csv("output/submission_v10.csv", index=False)
    print("Target value distribution in the Test data: ")
    print(submission["next_month_plan"].value_counts())


if __name__ == "__main__":
    main()
