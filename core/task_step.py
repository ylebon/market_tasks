import time

from logbook import Logger


class TaskStep(object):
    """Tasks Factory

    """

    def __init__(self, name):
        self.name = name
        self.log = Logger(self.log_name)

    def execute(self):
        """
        Execute task

        """
        start_time = time.time()
        self.run()
        duration = time.time() - start_time
        self.log.info("msg='task finished duration={}".format(duration))

    def run(self):
        """
        Run task

        """
        pass

    @property
    def log_name(self):
        return "[task] {}".format(self.name)
