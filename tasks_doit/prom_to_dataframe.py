import digdag
import joblib

from core.tasks.histdata import prom_to_dataframe


def run(prom_data):
    task = prom_to_dataframe.Task("prom_to_dataframe")
    dataframe = task.run(joblib.load(prom_data))
    output_file = "/tmp/digdag/output/dataframe.parquet"
    dataframe.to_parquet(output_file)
    digdag.env.store({"dataframe": output_file})