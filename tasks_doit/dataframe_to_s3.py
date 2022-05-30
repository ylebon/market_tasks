import os

from dask import dataframe as dd

from varatra_tasks.core.tasks.histdata import dataframe_to_s3


def run(dataframe):
    df = dd.read_parquet(dataframe)
    os.environ["PARQUET.S3_BUCKET_PATH"] = "s3://varatra-mkdata/parquet"
    task = dataframe_to_s3.Task("dataframe_to_s3")
    task.run(df)
