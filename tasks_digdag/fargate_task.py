import os
import sys

from logbook import StreamHandler

from varatra_tasks.core.tasks.aws import run_fargate_task

StreamHandler(sys.stdout, level=os.getenv('LOGLEVEL', 'INFO')).push_application()


def run(name, cluster, command, task_def, memory=1024, cpu=512):
    t = run_fargate_task.Task('fargate_task')
    response = t.run(name, cluster, command, memory=memory, cpu=cpu, task_def=task_def)
    print(response)
