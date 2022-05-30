from shutil import make_archive

from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Zip directory
    """

    def run(self, directory, zip_file=None):
        """Zip directory"""
        self.log.info("msg='zipping directory' dir={}".format(directory))
        if not zip_file:
            zip_file = directory + ".zip"
        make_archive(zip_file, 'zip', directory)
        return zip_file
