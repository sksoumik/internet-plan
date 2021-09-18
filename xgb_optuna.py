import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import optuna
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score
import functools


# read training and test data
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
        "age_group",
    ]
    X_encode = pd.get_dummies(train_data, columns=categorical_columns)

    return X_encode


# Split the data into training and testing sets
def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


# params for xgboost for multi-class classification
def xgb_params():
    params = {
        "objective": "multi:softprob",
        "num_class": 8,
        "booster": "gbtree",
        "max_depth": 8,
        "min_child_weight": 1,
        "gamma": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eta": 0.1,
        "seed": 42,
        "n_estimators": 100,
        "early_stopping_rounds": 10,
        "eval_metric": "mlogloss",
        "verbosity": 2,
        "n_jobs": -1,
        "random_state": 42,
        # max_delta_step
        "max_delta_step": 8,
    }
    return params


# xgboost model for multi-class classification
def xgboost_model(X_train, X_test, y_train, y_test, params):
    # xgboost model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


# calculate the confusion matrix
def confusion_matrix_cal(y_test, y_pred):
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm


# objective function for hyperparameter tuning
def objective(trial, X_train, X_test, y_train, y_test):
    # define the params for xgboost
    params = xgb_params()
    # define the learning rate
    params["eta"] = trial.suggest_uniform("eta", 0, 1)
    # define the max depth
    params["max_depth"] = trial.suggest_int("max_depth", 1, 100)
    # define the min child weight
    params["min_child_weight"] = trial.suggest_int("min_child_weight", 0, 100)
    # define the gamma
    params["gamma"] = trial.suggest_uniform("gamma", 0, 100)
    # define the subsample
    params["subsample"] = trial.suggest_uniform("subsample", 0, 1)
    # define the colsample_bytree
    params["colsample_bytree"] = trial.suggest_uniform("colsample_bytree", 0, 1)
    # max_delta_step
    params["max_delta_step"] = trial.suggest_int("max_delta_step", 1, 10)
    # define the seed
    params["seed"] = 42
    # define the number of classes
    params["num_class"] = 8
    # define the objective function
    params["objective"] = "multi:softprob"
    # define the booster
    params["booster"] = "gbtree"
    # define the number of rounds
    params["n_estimators"] = 100
    # define the early stopping round
    params["early_stopping_rounds"] = 10
    # define the metrics
    params["eval_metric"] = "mlogloss"
    # define the verbosity
    params["verbosity"] = 2
    # define the random state
    params["random_state"] = 42
    # define the number of folds
    params["n_jobs"] = -1
    # define the verbosity
    params["verbosity"] = 2
    # define the random state
    params["random_state"] = 42

    # train the xgboost model
    model, y_pred = xgboost_model(X_train, X_test, y_train, y_test, params)
    # calculate the confusion matrix
    # cm = confusion_matrix_cal(y_test, y_pred)
    # calculate the f1 score
    f1 = f1_score(y_test, y_pred, average="macro")
    return f1


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
    # train_data.drop(["age_group"], axis=1, inplace=True)
    # test_data.drop(["age_group"], axis=1, inplace=True)

    map_dict_age_grp = {
        "30-40": "30_40",
        "40-50": "40_50",
        "20-30": "20_30",
        "50-60": "50_60",
        "60-70": "60_70",
        ">70": "more_than_70",
        "<20": "less_than_20",
    }
    train_data["age_group"].replace(map_dict_age_grp, inplace=True)
    test_data["age_group"].replace(map_dict_age_grp, inplace=True)

    # select features
    X, y = select_features(train_data)
    # encode categorical columns
    X_encode = encode_categorical_columns(X)
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X_encode, y, test_size=0.2, random_state=42
    )

    # print(X_test)
    # xgboost model
    params = xgb_params()
    model, y_pred = xgboost_model(X_train, X_test, y_train, y_test, params)
    # confusion matrix
    cm = confusion_matrix_cal(y_test, y_pred)
    # calculate the f1 score
    f1_score_ = f1_score(y_test, y_pred, average="macro")
    print("Macro F1 Score")
    print(f1_score_)
    # save the model
    # model.save_model("model/xgb_model_v2.model")

    ######################### OPTUNA PARAMS ###############################

    # create a new optuna study for hyperparameter tuning
    study = optuna.create_study(direction="maximize")
    # define the objective function
    study.optimize(
        functools.partial(
            objective, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
        ),
        n_trials=100,
    )
    # print the best params
    print("\n================= Best Params ===================")
    print(study.best_params)
    # print the best value
    print("Best value:")
    print(study.best_value)

    print("Best score: ", study.best_value)

    ######################### TEST DATA ###############################

    test_data = encode_categorical_columns(test_data)

    test_data = test_data[X_train.columns]
    # check the model using the test_data
    y_pred_test = model.predict(test_data)
    print(y_pred_test)
    # save the y_pred_test in kaggle format
    submission = pd.DataFrame(
        {
            "primary_identifier": test_data["primary_identifier"],
            "next_month_plan": y_pred_test,
        }
    )
    # submission.to_csv("output/submission_v2.csv", index=False)
    print("Target value distribution in the Test data: ")
    print(submission["next_month_plan"].value_counts())


if __name__ == "__main__":
    main()
