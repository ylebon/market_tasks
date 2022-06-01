import os
import tempfile

import digdag
from dask import dataframe as dd

from core.histdata import dataframe_to_parquet


def run(dataframe):
    output_dir = tempfile.mkdtemp()
    os.environ['PARQUET.LOCAL_DIRECTORY'] = output_dir
    df = dd.read_parquet(dataframe)
    task = dataframe_to_parquet.Task("dataframe_to_parquet")
    task.run(df)
    digdag.env.store({"parquet_directory": output_dir})
