import json
import os
import tempfile

import boto3
import dateutil.parser
from joblib import load

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep
from tasks.core.tasks.features import load_parquet
from tasks.core.tasks.features import sync_s3
from tasks.core.tasks.model import classification_report


class Task(TaskStep):
    """
    Sync S3

    """
    s3 = None
    os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

    def run(self, model_key):

        services_config = ConfigServices.create()

        if not self.s3:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=services_config.get_value("AMAZON.ACCESS_KEY"),
                aws_secret_access_key=services_config.get_value("AMAZON.SECRET_KEY"),
            )

        # bucket name
        bucket_name = services_config.get_value("TESTING_ML.S3_BUCKET")
        model_local = services_config.get_value("MODEL.LOCAL_DIRECTORY")

        model_file = os.path.join(model_key, 'model.joblib')
        meta_file = os.path.join(model_key, 'meta.json')
        if not os.path.exists(meta_file):
            self.log.error(f"meta file not found: {meta_file}")
            return
        else:
            with open(meta_file, 'r') as fr:
                try:
                    meta = json.loads(fr.read())
                except json.decoder.JSONDecodeError as error:
                    self.log.error(f"failed to decode json file '{meta_file}' error='{error}'")
                    return
                features = meta["features"]
                target = meta["target"]
                dataset = meta["dataset"]

                instrument_id = meta["instrument_id"]
                # sync features parquet
                t_sync_s3 = sync_s3.Task("sync_s3")
                exchange, symbol = instrument_id.split("_", 1)
                parquet_filter = {'exchange': exchange, 'symbol': symbol}
                t_sync_s3.run(parquet_filter=parquet_filter)

                # load features parquet
                t_load_parquet = load_parquet.Task("load_parquet")
                df = t_load_parquet.run(instrument_id)
                df.dropna(inplace=True)
                df = df[dateutil.parser.parse(dataset['end_date']) < df.index]

                # create X, y
                if len(df.index) < 2:
                    self.log.error("not enough data for testing")
                    return
                self.log.info(
                    f"msg='testing model' model='{model_key}' target='{target}' start_date='{df.index[0]}' end_date='{df.index[-1]}'"
                )
                X = df[features]
                y = df[target]

                model = load(model_file)
                t = classification_report.Task("classification_report")
                testing_result = t.run(model, X, y)

                result = {
                    'dataset': {
                        'start_date': str(df.index[0]),
                        'end_date': str(df.index[-1])
                    },
                    'result': testing_result.to_dict()
                }

                testing_file = os.path.join(tempfile.mkdtemp(), 'testing.json')
                testing_key = os.path.relpath(model_local, model_key) + "/testing.json"

                with open(testing_file, 'w') as fw:
                    json.dump(result, fw, default=str)

                self.s3.upload_file(testing_file, bucket_name, testing_key)


if __name__ == "__main__":
    t = Task("script_to_s3")
    t.run("OANDA_GBP_USD")
