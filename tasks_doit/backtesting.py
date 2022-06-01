import os
import sys

from logbook import StreamHandler

from backtesting.core.backesting import SignalBackTesting
from patterns.recursionlimit import recursionlimit
from core.backtesting import upload_report
from core.minio import dowload_dir
from core.minio import upload_file
from utils.time_util import get_last_interval

StreamHandler(sys.stdout, level=os.getenv('LOG_LEVEL', 'INFO')).push_application()


def run(signaler_name, instrument, start_date, end_date, bucket_name, bucket_key, last_interval=0):
    # last interval
    if last_interval:
        start_date, end_date = get_last_interval(last_interval)

    # backtester
    backtester = SignalBackTesting.create_from_name(
        signaler_name,
        [instrument],
        '/tmp',
        simulation=None,
        ordering=None,
        signal_plot=False,
        monitoring=False
    )

    # run backtester
    with recursionlimit(150000):
        backtester.start(
            start_date,
            end_date,
            from_db=True,
        )

    # backtesting report
    output_file = f'/tmp/{signaler_name}_{instrument}.csv'
    perf = backtester.get_perf()
    df = perf.to_df()
    df['start_date'] = start_date
    df['end_date'] = end_date
    df.to_csv(output_file)
    # digdag.env.store({"backtesting_report": output_file})

    # minio upload file
    task = upload_file.Task("upload_file")
    task.run(bucket_name, output_file, bucket_key)


def report(bucket_name):
    tmp_output_dir = "/tmp/output"
    # task download dir
    t_download_dir = dowload_dir.Task("download_dir")
    t_download_dir.run(bucket_name, tmp_output_dir)
    # task upload report
    t_upload_report = upload_report.Task("upload_report")

    # filter csv files
    csv_files = [os.path.join(tmp_output_dir, x) for x in os.listdir(tmp_output_dir) if
                 os.path.splitext(x)[1] == '.csv']
    t_upload_report.run(csv_files)


if __name__ == "__main__":
    run("bbands.percentile", "BITSTAMP_XRP_USD", "2020-01-02", "2020-01-02", last_interval=24)
