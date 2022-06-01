import os
import sys

from logbook import StreamHandler

from core.tasks.scaleway import run_scaleway_task

StreamHandler(sys.stdout, level=os.getenv('LOGLEVEL', 'INFO')).push_application()


def run(name, image, command, memory=1024, cpu=512):
    t = run_scaleway_task.Task('scaleway_task')
    response = t.run(name, image, command, memory=memory, cpu=cpu)
    print(response)
