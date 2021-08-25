MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[[GB] [London] [izzyweber] TaxiFare + V1]"
BUCKET_NAME = 'wagon-data-699-weber'
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'
STORAGE_LOCATION = 'models/TaxiFareModel/model.joblib'

params = dict(nrows=10000,
              upload=True,
              # set to False to get data from GCP (Storage or BigQuery)
              local=False,
              gridsearch=False,
              optimize=True,
              estimator="xgboost",
              #mlflow=True,  # set to True to log params to mlflow
              #experiment_name=experiment,
              pipeline_memory=None,  # None if no caching and True if caching expected
              distance_type="manhattan",
              feateng=["distance_to_center", "direction",
                       "distance", "time_features", "geohash"],
              n_jobs=-1)  # Try with njobs=1 and njobs = -1
