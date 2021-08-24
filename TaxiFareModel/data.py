import pandas as pd
from sklearn.model_selection import train_test_split

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    return df


def airport_trip(row):
    if (round(row.pickup_latitude, 2) == 40.77) | (round(row.pickup_longitude, 2) == -73.872):
        return "pu_laguardia"
    elif (round(row.pickup_latitude, 2) == 40.64) | (round(row.pickup_latitude, 2) == -73.77):
        return "pu_jfk"
    elif (round(row.dropoff_latitude, 2) == 40.77) | (round(row.dropoff_latitude, 2) == -73.87):
        return "do_laguardia"
    elif (round(row.dropoff_latitude, 2) == 40.64) | (round(row.dropoff_latitude, 2) == -73.77):
        return "do_jfk"
    else:
        return "non_airport_trip"

def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    df["airport_trip"] = df.apply(airport_trip, axis=1)
    return df

def holdout(df):
    y = df.pop("fare_amount")
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    return X, y, X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = get_data()
