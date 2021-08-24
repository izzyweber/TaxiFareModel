from ml_flow_test import EXPERIMENT_NAME
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import numpy as np

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data, holdout
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

class Trainer():
    MLFLOW_URI = "https://mlflow.lewagon.co/"
    EXPERIMENT_NAME = "[[GB] [London] [izzyweber] TaxiFare + V1]"

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

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
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
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

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    X, y, X_train, X_test, y_train, y_test = holdout(df)
    trainer = Trainer(X, y)
    trainer.run(X_train, y_train)
    trainer.mlflow_log_param("Estimator", "Linear")
    trainer.mlflow_log_metric("RMFSE", trainer.evaluate(X_test, y_test))
    print(f'Testing complete. Your RMSE was:{trainer.evaluate(X_test, y_test)}')
