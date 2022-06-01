import json
import os
import tempfile
from datetime import datetime

import boto3

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Deploy TPOT script to S3

    """
    s3 = None
    os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

    def run(self, instrument_id, target, model_file, meta=None):
        services_config = ConfigServices.create()

        if not self.s3:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=services_config.get_value("AMAZON.ACCESS_KEY"),
                aws_secret_access_key=services_config.get_value("AMAZON.SECRET_KEY"),
            )

        bucket_name = services_config.get_value("TPOT.S3_BUCKET")
        model_key = instrument_id + "/" + target + "/" + os.path.basename(model_file)
        meta_key = instrument_id + "/" + target + "/" + "meta.json"

        # upload model
        self.s3.upload_file(model_file, bucket_name, model_key)

        # upload meta
        meta = meta or dict(uploaded_at=datetime.now())
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(str(meta).encode())
            self.s3.upload_file(fp.name, bucket_name, meta_key)


if __name__ == "__main__":
    t = Task("script_to_s3")
    t.run("OANDA_EUR_USD", "profit__binary__10M",
          "/Users/madazone/Workspace/varatra/signaler/feeds/tests/test_replay.py")
