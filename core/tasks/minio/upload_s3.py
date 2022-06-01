import os

from minio import Minio
from minio.error import ResponseError

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Upload file from S3

    """
    s3 = None

    def run(self, bucket_name, local_src, remote_dst=None):
        """
        Upload from s3

        """

        if not self.__class__.s3:
            self.__class__.s3 = Minio('s3.amazonaws.com',
                                      access_key=ConfigServices.create().get_value("AMAZON.ACCESS_KEY"),
                                      secret_key=ConfigServices.create().get_value("AMAZON.SECRET_KEY"),
                                      secure=True
                                      )
        try:
            if not os.path.exists(local_src):
                self.log.error(f"msg='fail to find the local file' err={local_src}")
            else:
                pass

            remote_dst = remote_dst or os.path.basename(local_src)
            self.log.info(f"msg='uploading file to s3' bucket='{bucket_name}' src='{local_src}' dst='{remote_dst}'")
            data = self.s3.fput_object(bucket_name, remote_dst, local_src)
            return remote_dst
        except ResponseError as err:
            self.log.error(f"msg='fail to download file' err={err}")

        return remote_dst
