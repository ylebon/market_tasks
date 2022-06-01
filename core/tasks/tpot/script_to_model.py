import os

from core.task_step import TaskStep


class Task(TaskStep):
    """
    TPOT script to model

    """
    s3 = None
    os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

    def run(self, script, dataset):
        pass


if __name__ == "__main__":
    t = Task("script_to_s3")
    script = "/Users/madazone/Workspace/varatra/signaler/tasks/launchers/tpot_classifier_pipeline.py"
    t.run(script, dataset)
