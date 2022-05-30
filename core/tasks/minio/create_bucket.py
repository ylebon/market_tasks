from minio import Minio, error

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Download file from S3

    """
    client = None

    def run(self, bucket_name, ignore_exists=True):
        """
        Create bucket

        """
        services_config = ConfigServices.create()
        self.client = Minio(services_config.get_value("MINIO.URL"),
                            access_key=services_config.get_value("MINIO.ACCESS_KEY"),
                            secret_key=services_config.get_value("MINIO.SECRET_KEY"),
                            secure=False
                            )

        try:
            self.client.make_bucket(bucket_name)
        except error.BucketAlreadyOwnedByYou:
            if not ignore_exists:
                raise


if __name__ == "__main__":
    t = Task("create_client")
    t.run("oks")
