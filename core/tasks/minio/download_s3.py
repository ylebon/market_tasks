import os

from minio import Minio
from minio.error import ResponseError

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Download file from S3

    """
    s3 = None

    def run(self, bucket_name, remote_src, local_dst):
        """
        Download from s3

        """

        if not self.__class__.s3:
            self.__class__.s3 = Minio('s3.amazonaws.com',
                                      access_key=ConfigServices.create().get_value("AMAZON.ACCESS_KEY"),
                                      secret_key=ConfigServices.create().get_value("AMAZON.SECRET_KEY"),
                                      secure=True
                                      )
        try:
            if os.path.isdir(local_dst):
                local_dst = os.path.join(local_dst, os.path.basename(remote_src))
            local_dst = os.path.normpath(local_dst)
            self.log.info(f"msg='downloading s3 file' bucket='{bucket_name}' src='{remote_src}' dst='{local_dst}'")
            data = self.s3.fget_object(bucket_name, remote_src, local_dst)
            return local_dst
        except ResponseError as err:
            self.log.error(f"msg='fail to download file' err={err}")

        return local_dst
