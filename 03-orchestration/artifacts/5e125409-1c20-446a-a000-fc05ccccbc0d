import pickle
import pandas as pd
import datetime as dt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow
from prefect import task, flow, get_run_logger
from prefect.task_runners import SequentialTaskRunner

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('nyc-taxi-experiment')
mlflow.sklearn.autolog()

def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical, date):

    with mlflow.start_run():

        logger = get_run_logger()
        train_dicts = df[categorical].to_dict(orient='records')
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts) 
        y_train = df.duration.values

        logger.info(f"The shape of X_train is {X_train.shape}")
        logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        mse = mean_squared_error(y_train, y_pred, squared=False)
        logger.info(f"The MSE of training is: {mse}")

        with open(f"./artifacts/dv-{date}.b", "wb") as f_out:
                    pickle.dump(dv, f_out)
        mlflow.log_artifact(f"./artifacts/dv-{date}.b", artifact_path = "preprocessor")
        mlflow.sklearn.log_model(f"./artifacts/model-{date}.bin", artifact_path = 'model')

    return lr, dv

@task
def run_model(df, categorical, dv, lr):

    logger = get_run_logger()    
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date = None):

    logger = get_run_logger()
    if date is None:
        logger.info(f'Init date is set to None. Setting to default')
        train_yyyy_mm = f'{dt.datetime.today().year}-{str(dt.datetime.today().month - 2).zfill(2)}'
        test_yyyy_mm = f'{dt.datetime.today().year}-{str(dt.datetime.today().month - 1).zfill(2)}'
    else:
        logger.info(f'Init date is set to {date}')        
        date = pd.to_datetime(date)
        train_yyyy_mm = f'{date.year}-{str(date.month - 2).zfill(2)}'
        test_yyyy_mm = f'{date.year}-{str(date.month - 1).zfill(2)}'

    train_path = f'./data/fhv_tripdata_{train_yyyy_mm}.parquet'
    test_path = f'./data/fhv_tripdata_{test_yyyy_mm}.parquet'
    logger.info(f'Train path: {train_path} & valid path: {test_path}')        

    return train_path, test_path

@flow(task_runner=SequentialTaskRunner())
def main(date="2021-03-15"):

    categorical = ['PUlocationID', 'DOlocationID']
    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)#.result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)#.result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical, date).result()
    run_model(df_val_processed, categorical, dv, lr)


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name="cron-schedule-train",
    schedule=CronSchedule(
        cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
)
