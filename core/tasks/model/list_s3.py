import fnmatch
import os

import boto3

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    List files

    """
    s3 = None
    os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

    def run(self, instrument_id: str = "*", model_target: str = "*", model_type="*", run_id="*", force=False):
        services_config = ConfigServices.create()

        if not self.s3:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=services_config.get_value("AMAZON.ACCESS_KEY"),
                aws_secret_access_key=services_config.get_value("AMAZON.SECRET_KEY"),
            )

        # bucket name
        bucket_name = services_config.get_value('MODEL.S3_BUCKET').replace("s3://", "")

        # list objects
        paginator = self.s3.get_paginator("list_objects")
        page_iterator = paginator.paginate(Bucket=bucket_name)

        # create files to sync
        files_list = list()
        for page in page_iterator:
            if "Contents" in page:
                for key in page["Contents"]:
                    file_key = key["Key"]
                    files_list.append(file_key)
        self.log.info(f"msg='total number of files in S3' total='{len(files_list)}'")

        # Filter parquet
        parquet_filter = dict(run_id=run_id, model_target=model_target, model_type=model_type)
        if instrument_id != "*":
            exchange, symbol = instrument_id.split("_", 1)
        else:
            exchange, symbol = "*", "*"
        model_target = parquet_filter.get("model_target", "*")
        model_type = parquet_filter.get("model_type", "*")
        run_id = parquet_filter.get("run_id", "*")

        pattern = f"exchange={exchange}/symbol={symbol}/target={model_target}/type={model_type.upper()}/run_id={run_id}/*"

        matching_files = fnmatch.filter(files_list, pattern)
        self.log.info(f"msg='total maching files in S3' bucket='{bucket_name}' pattern='{pattern}' total='{len(matching_files)}'")

        # return matching files
        return matching_files


if __name__ == "__main__":
    from logbook import StreamHandler
    import sys

    StreamHandler(sys.stdout).push_application()
    t = Task("script_to_s3")
    r = t.run()
    print(r)
