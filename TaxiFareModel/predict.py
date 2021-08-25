import os
from math import sqrt

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from TaxiFareModel.data import holdout, clean_data, df_optimized, get_data
PATH_TO_LOCAL_MODEL = 'model.joblib'


def get_test_data():
    """method to get the training data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    print("helloooww")
    path = "../raw_data/test.csv"
    df = pd.read_csv(path)
    df["fare_amount"]=0
    df = df[['key', 'fare_amount', 'pickup_datetime', 'pickup_longitude',
             'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
             'passenger_count']]
    return df


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


def predict():
    df = get_data(nrows=100_000)
    df = clean_data(df)
    df = df_optimized(df)
    X, y, X_train, X_test, y_train, y_test = holdout(df)
    pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(X_test)
    else:
        y_pred = pipeline.predict(X_test)
    # X_test["fare_amount"] = y_pred
    # df_sample = X_test[["key", "fare_amount"]]
    # name = f"predictions_test_ex.csv"
    # df_sample.to_csv(name, index=False)
    print(evaluate_model(y_test, y_pred))
    return y_pred

if __name__ == '__main__':
    y_pred = predict()

