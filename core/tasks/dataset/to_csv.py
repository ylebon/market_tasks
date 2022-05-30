import os
import toml
from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Return orange datasets

    """

    def run(self, name, dataset, meta=None):
        """Orange datasets directory"""
        datasets_directory = ConfigServices.create().get_value("DATASETS.DIRECTORY")
        meta = dict() if meta is None else meta
        # create directory
        if not os.path.exists(datasets_directory):
            os.makedirs(datasets_directory)
        # create CSV file
        csv_file = os.path.join(datasets_directory, f"{name}.csv")
        self.log.info("msg_file='create dataset' output='{0}'".format(csv_file))
        dataset.to_csv(csv_file)
        # create
        meta_file = os.path.join(datasets_directory, f"{name}.meta")
        with open(meta_file, 'w') as fw:
            self.log.info("msg_file='create meta' output='{0}'".format(meta_file))
            toml.dump(meta, fw)

