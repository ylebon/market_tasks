import joblib

from varatra_tasks.core.tasks.histdata import load_prom
from varatra_tasks.core.tasks.minio import upload_file
from varatra_utils.time_util import get_date


def run(instrument, start_date, end_date, destination, bucket_name="digdag"):
    t = load_prom.Task("load_prom")
    start_date = get_date(start_date)
    end_date = get_date(end_date)

    data = t.run(instrument, start_date, end_date)

    # prom data
    output_file = "/tmp/prom_data.joblib"
    joblib.dump(data, output_file)

    # upload to minio
    task = upload_file.Task("upload_file")
    task.run(bucket_name, output_file, destination)
