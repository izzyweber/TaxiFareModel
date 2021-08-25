#from ml_flow_test import EXPERIMENT_NAME
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
import numpy as np


from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import airport_trip, get_data, clean_data, holdout, df_optimized
import TaxiFareModel.params as params

# import mlflow
# from mlflow.tracking import MlflowClient
# from memoized_property import memoized_property

import joblib
from google.cloud import storage

class Trainer():
    def __init__(self, X, y, model):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.model = model
        self.pipeline = None
        self.X = X
        self.y = y
        #self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        cat_pipe = Pipeline([
            ("ohe", OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime']),
            ("cats", cat_pipe, ["airport_trip"])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', self.model)
        ])
        self.pipeline = pipe
        return pipe

    def run(self, X_train, y_train):
        """set and train the pipeline"""
        self.pipeline=self.set_pipeline()
        self.pipeline = self.pipeline.fit(X_train, y_train)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return np.sqrt(((y_pred - y_test)**2).mean())

    # @memoized_property
    # def mlflow_client(self):
    #     mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
    #     return MlflowClient()

    # @memoized_property
    # def mlflow_experiment_id(self):
    #     try:
    #         return self.mlflow_client.create_experiment(self.experiment_name)
    #     except BaseException:
    #         return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    # @memoized_property
    # def mlflow_run(self):
    #     return self.mlflow_client.create_run(self.mlflow_experiment_id)

    # def mlflow_log_param(self, key, value):
    #     self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    # def mlflow_log_metric(self, key, value):
    #     self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def upload_model_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(params.BUCKET_NAME)
        blob = bucket.blob(params.STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')

    def save_model(self, reg):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(reg, 'model.joblib')
        print("saved model.joblib locally")

        # Implement here
        self.upload_model_to_gcp()
        print(f"uploaded model.joblib to gcp cloud storage under \n => {params.STORAGE_LOCATION}")

if __name__ == "__main__":
    df = get_data(nrows=100_000)
    df = clean_data(df)
    df = df_optimized(df)
    X, y, X_train, X_test, y_train, y_test = holdout(df)
    print(X_train.columns)
    for model in [LinearRegression(), Lasso(), ElasticNet(), Ridge()]:
        trainer = Trainer(X, y, model)
        model = trainer.run(X_train, y_train)
        #trainer.mlflow_log_param("Estimator", model)
        #trainer.mlflow_log_metric("RMSE", trainer.evaluate(X_test, y_test))
        trainer.save_model(model)
        print(f'Testing complete for {model}. Your RMSE was:{trainer.evaluate(X_test, y_test)}')
