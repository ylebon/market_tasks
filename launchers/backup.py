import os
import sys

from logbook import StreamHandler

StreamHandler(sys.stdout, level=os.getenv('LOG_LEVEL', 'INFO')).push_application()
import os

from dask import dataframe as dd

from core.tasks.histdata import dataframe_to_s3
from core.tasks.histdata import load_prom
from core.tasks.histdata import prom_to_dataframe
from utils.time_util import get_date


def launch(instrument_id, start_date, end_date):
    """
    Backup prom data to S3

    """
    # load prom
    t_load_prom = load_prom.Task("load_prom")
    start_date = get_date(start_date)
    end_date = get_date(end_date)
    data = t_load_prom.run(instrument_id, start_date, end_date)

    # upload to minio
    t_prom_to_dataframe = prom_to_dataframe.Task("prom_to_dataframe")
    dataframe = t_prom_to_dataframe.run(data)
    output_file = "/tmp/dataframe.parquet"
    dataframe.to_parquet(output_file, engine='pyarrow')

    # dataframe to s3
    df = dd.read_parquet(output_file)
    os.environ["PARQUET.S3_BUCKET_PATH"] = "s3://feeds-parquet"
    t_dataframe_to_s3 = dataframe_to_s3.Task("dataframe_to_s3")
    t_dataframe_to_s3.run(df)
