import os
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3.query import Query

import pandas as pd
import numpy as np
from pandas import DataFrame
import time

from IPython import embed


class CloudMetrics(object):
    ''' Query metrics from cloud monitoring '''

    def __init__(self, project_name, req_names):
        # Setup client and query to cloud monitoring.
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = project_name
        self.req_names = req_names # Operations that should be considered part of the context (list)


    def get_request_count(self, minutes=10, pod_filter='frontend'):

        # Run a query for the request count.
        query = Query(self.client, project=self.project_name, metric_type='istio.io/service/server/request_count', minutes=minutes)
        df = query.as_dataframe(label='pod_name')

        # Filter columns that pertain to the pods we are interested in.
        columns_list = df.columns.tolist()
        filtered_pods = [column[0]for column in columns_list if pod_filter in column[0]]
        if len(filtered_pods) == 0:
             return 0

        # Construct dataframe of rps each minute.
        df = df[list(set(filtered_pods))]
        df = df.resample('T').sum()
        df = df.sum(axis=1)
        df /= 60.0
        df = df.diff()
        print(df)

        return df.iloc[-1]

    def get_request_df(self, minutes=10, pod_filter='frontend'):

        # Run a query for the request count.
        query = Query(self.client, project=self.project_name, metric_type='istio.io/service/server/request_count', minutes=minutes)
        df = query.as_dataframe(label='pod_name')

        # Filter columns that pertain to the pods we are interested in.
        columns_list = df.columns.tolist()
        filtered_pods = [column[0]for column in columns_list if pod_filter in column[0]]
        # if len(filtered_pods) == 0:
        #     return 0

        # Construct dataframe of rps each minute.
        df = df[list(set(filtered_pods))]
        df = df.resample('T').sum()
        df = df.sum(axis=1)
        df /= 60.0
        df = df.diff()
        print(df)

        return df


    def get_request_dist(self, minutes=10):
        
        if self.req_names is None or len(self.req_names) == 0:
            return [], []

        req_names = self.req_names
        query = Query(self.client, project=self.project_name, metric_type='istio.io/service/server/request_count', minutes=minutes)
        df = query.as_dataframe('request_operation')

        # Query the rps for each of the operations we are considering.
        columns_list = df.columns.tolist()

        req_index, req_vector = [], np.zeros(len(req_names))
        for i, req_name in enumerate(sorted(req_names)):

            # Construct a minute level timeseries for the request operation.
            df_cols = [x[0]for x in columns_list if req_name in x[0]]
            req_df = df[list(set(df_cols))]
            req_df = req_df.resample('T').sum()
            req_df = req_df.sum(axis=1)
            req_df /= 60.0
            req_df = req_df.diff()

            # Store the most recent minute in the a dictionary and vector of results.
            req_index.append(req_name)
            req_vector[i] = req_df.iloc[-1]
        
        req_vector = req_vector / req_vector.sum()
        return req_vector, req_index

