import os
import tarfile
import toml
import shutil
import joblib

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """Create model archive

    """
    model_local_directory = ConfigServices.create().get_value("MODEL.LOCAL_DIRECTORY")

    def run(self, pipeline, model, meta=None):
        self.log.info("msg='dumping model to directory' directory={0}".format(self.__class__.model_local_directory))
        model_directory = os.path.join(self.__class__.model_local_directory, pipeline.name)
        meta = dict() if meta is None else meta
        gzip_file = model_directory + ".gz"

        # create directory
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        if os.path.exists(gzip_file):
            os.remove(gzip_file)

        # create model
        model_file = os.path.join(model_directory, 'model.model')
        joblib.dump(model, model_file)

        # create params
        meta_file = os.path.join(model_directory, 'model.meta')
        with open(meta_file, "w") as fw:
            toml.dump(meta, fw)

        # compress directory
        with tarfile.open(gzip_file, "w:gz") as tar:
            tar.add(model_directory, arcname=os.path.basename(model_directory))

        # clean directory
        if os.path.exists(gzip_file):
            shutil.rmtree(model_directory)
