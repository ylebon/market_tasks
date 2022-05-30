import json
import os
import shutil
import tempfile
from datetime import datetime

import boto3

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Deploy TPOT script to S3

    """
    s3 = None
    os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

    def run(self, instrument_id: str, model_features, model_target: str, model_type, model_category, model_file,
            meta=None, zip_dir=None, nb_file=None):
        services_config = ConfigServices.create()
        self.log.info(f"msg='dumping model to S3' model_type='{model_type}' model_category='{model_category}'")

        if not self.s3:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=services_config.get_value("AMAZON.ACCESS_KEY"),
                aws_secret_access_key=services_config.get_value("AMAZON.SECRET_KEY"),
            )

        # bucket name
        bucket_name = services_config.get_value("MODEL.S3_BUCKET")

        # model
        exchange, symbol = instrument_id.split("_", 1)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_key = f"exchange={exchange}/symbol={symbol}/target={model_target}/type={model_type.upper()}/run_id={run_id}/"
        model_key = dir_key + "model.joblib"
        # upload model
        self.s3.upload_file(model_file, bucket_name, model_key)

        # zip dir
        if zip_dir:
            model_zip = f"/tmp/model.zip"
            if os.path.exists(model_zip):
                self.log.info("msg='deleting previous model.zip'")
                os.remove(model_zip)
            self.log.info(f"msg='archiving model' path='{zip_dir}'")
            shutil.make_archive("/tmp/model", "zip", zip_dir, ".")
            zip_key = dir_key + "model.zip"
            self.log.info(
                f"msg='uploading model zip to S3' model_file='{model_zip}' bucket_name='{bucket_name}' s3_key='{zip_key}'")
            self.s3.upload_file(model_zip, bucket_name, zip_key)

        # if notebook
        if nb_file:
            nb_key = dir_key + "model.ipynb"
            self.log.info(
                f"msg='uploading model notebook to S3' model_file='{nb_file}' bucket_name='{bucket_name}' s3_key='{nb_key}'")
            self.s3.upload_file(nb_file, bucket_name, nb_key)

        # upload meta
        meta_key = dir_key + "meta.json"
        meta = meta or dict()
        meta['zip'] = zip_dir is not None
        meta['uploaded_at'] = datetime.utcnow()
        meta['model_type'] = model_type
        meta['instrument_id'] = instrument_id
        meta['features'] = list(model_features)
        meta['target'] = model_target
        meta['size'] = os.stat(model_file).st_size
        meta['category'] = model_category
        meta_file = os.path.join(tempfile.mkdtemp(), 'meta.json')
        with open(meta_file, 'w') as fw:
            json.dump(meta, fw, default=str)
        self.log.info(
            f"msg='uploading model meta to S3' meta_file='{meta_file}' bucket_name='{bucket_name}' s3_key='{meta_key}'"
        )
        self.s3.upload_file(meta_file, bucket_name, meta_key)


if __name__ == "__main__":
    t = Task("script_to_s3")
    t.run("OANDA_EUR_USD", 'oko', "profit__binary__10M", "tpot", "regressor"
                                                                 "/Users/madazone/Workspace/varatra/signaler/feeds/tests/test_replay.py")
