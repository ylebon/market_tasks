from minio import Minio

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Download directory content

    """
    client = None

    def run(self, bucket_name, output_directory):
        """
        Create bucket

        """
        services_config = ConfigServices.create()
        self.client = Minio(services_config.get_value("MINIO.URL"),
                            access_key=services_config.get_value("MINIO.ACCESS_KEY"),
                            secret_key=services_config.get_value("MINIO.SECRET_KEY"),
                            secure=False
                            )

        for o in self.client.list_objects(bucket_name):
            if not o.is_dir:
                local_dst = f"{output_directory}/{o.object_name}"
                self.log.info(f"msg='downloading  file' bucket='{bucket_name}' src='{o.object_name}' dst='{local_dst}'")
                self.client.fget_object(bucket_name, o.object_name, local_dst)


if __name__ == "__main__":
    t = Task("download_directory")
    r = t.run("digdag", "output")
