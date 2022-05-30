import os
import toml

from tasks.core.task_step import TaskStep


class Task(TaskStep):

    @property
    def configs_directory(self):
       return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'mlearn'))

    def run(self, config_name):
        self.log.info("msg='read mlearn configuration file'")
        if not os.path.isfile(config_name):
            config_file = os.path.join(self.configs_directory, '{}.toml'.format(config_name))
        else:
            config_file = config_name
        config_date = toml.load(config_file)
        return config_date
