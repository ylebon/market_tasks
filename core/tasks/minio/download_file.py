from minio import Minio, error

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Upload file

    """
    client = None

    def run(self, bucket_name, source, destination, ignore_exists=True):
        """
        Create bucket

        """
        services_config = ConfigServices.create()
        self.client = Minio(services_config.get_value("MINIO.URL"),
                            access_key=services_config.get_value("MINIO.ACCESS_KEY"),
                            secret_key=services_config.get_value("MINIO.SECRET_KEY"),
                            secure=False
                            )

        return self.client.fget_object(bucket_name, source, destination)


if __name__ == "__main__":
    t = Task("download_file")
    t.run("digdag", "a/b", "/Users/madazone/Workspace/signaler/tasks/core/tasks/minio/download_s3.psy")
