#!/usr/bin/env python
# coding: utf-8

#get_ipython().system('pip freeze | grep scikit-learn')

import sys
import pickle
import numpy as np
import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def run():

    year = int(sys.argv[1])
    month = int(sys.argv[2])

    data_path = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_path =  f'./{year:04d}-{month:02d}-preds.parquet'

    df = read_data(data_path)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['preds'] = y_pred
    print(f'========== Mean preds: {np.mean(y_pred)}')

"""
    df[['ride_id', 'preds']].to_parquet(
        output_path,
        engine='pyarrow',
        compression=None,
        index=False
    )
"""

if __name__ == '__main__':
    run()