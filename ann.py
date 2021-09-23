import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

# load datasets
def read_data(train_data, test_data):
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)
    return train_data, test_data


# remove unnecessary columns
def remove_columns(train_data, test_data):
    rm_columns = [
        "device_type",
        "device_category",
        "gender",
        "district_name",
        "age_group",
        "vusage_onnet_avg",
        "vusage_offnet_avg",
        "add_on_tot_rental",
        "add_on_count",
        "dusage_avg",
    ]
    train_data.drop(rm_columns, axis=1, inplace=True)
    test_data.drop(rm_columns, axis=1, inplace=True)
    return train_data, test_data


# convert target variable to categorical
def convert_target_to_categorical(train_data):
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
    return train_data


# make train_data and test_data as numpy array
def make_train_val(train_data):
    drop_cols = ["next_month_plan", "primary_identifier"]
    X = train_data.drop(drop_cols, axis=1).copy()
    y = train_data["next_month_plan"].copy()
    return X, y


# normalize features with MinMaxScaler
def normalize_features(X, y):
    # Normalize features within range 0 (minimum) and 1 (maximum)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)
    Y = pd.get_dummies(y)
    X = X.values
    Y = Y.values
    return X, Y


# normalize test_data
def normalize_test_data(test_data):
    test_data_ = test_data.drop(["primary_identifier"], axis=1).copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    test_data_ = scaler.fit_transform(test_data_)
    test_data_ = pd.DataFrame(test_data_)
    return test_data_


# split training data into training and validation set using stratify
def split_training_data(X, Y):
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    return X_train, X_val, Y_train, Y_val


def define_model(X_train, Y_train):
    model = Sequential()
    model.add(Dense(units=164, activation="relu", input_dim=X_train.shape[1]))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=8, activation="relu"))
    model.add(Dense(units=Y_train.shape[1], activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


# train model
def train_model(model, X_train, Y_train, X_val, Y_val):
    model.fit(
        X_train,
        Y_train,
        epochs=40,
        batch_size=4,
        verbose=2,
        validation_data=(X_val, Y_val),
    )
    return model


# calculate the f1 score
def calculate_f1_score(model, X_val, Y_val):
    Y_pred = model.predict(X_val)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_val, axis=1)
    f1_score_ = f1_score(Y_true, Y_pred, average="macro")
    return f1_score_


# predict test_data
def predict_test_data(model, test_data):
    test_data = normalize_test_data(test_data)
    Y_pred = model.predict(test_data)
    Y_pred = np.argmax(Y_pred, axis=1)
    return Y_pred


def main():
    train_data, test_data = read_data(
        "input/train_dataset.csv", "input/test_dataset.csv"
    )
    train_data, test_data = remove_columns(train_data, test_data)
    train_data = convert_target_to_categorical(train_data)
    X, Y = make_train_val(train_data)
    X, Y = normalize_features(X, Y)
    X_train, X_val, Y_train, Y_val = split_training_data(X, Y)
    model = define_model(X_train, Y_train)
    model = train_model(model, X_train, Y_train, X_val, Y_val)
    f1_score_ = calculate_f1_score(model, X_val, Y_val)
    print("f1_score_: ", f1_score_)
    Y_pred = predict_test_data(model, test_data)
    # save the predictions in kaggle format
    submission = pd.DataFrame(
        {
            "primary_identifier": test_data["primary_identifier"],
            "next_month_plan": Y_pred,
        }
    )
    submission.to_csv("output/submission_v6.csv", index=False)


if __name__ == "__main__":
    main()