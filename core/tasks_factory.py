import importlib.util
import os

from patterns.singleton import MetaClassSingleton
from tasks.core.task_group import TaskGroup


class TasksFactory(metaclass=MetaClassSingleton):
    """
    Tasks Factory

    """
    tasks_directory = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        'tasks'
    ))

    def __init__(self):
        pass

    @classmethod
    def create(cls):
        """
        Create tasks

        Loop over all the directories
        """
        tasks_factory = TasksFactory()
        tasks_directory = cls.tasks_directory
        for directory in os.listdir(tasks_directory):
            if os.path.isdir(os.path.join(tasks_directory, directory)):
                tasks_group_directory = os.path.join(tasks_directory, directory)
                tasks_group = TaskGroup()
                for module in os.listdir(tasks_group_directory):
                    if '__init__' not in module and module.endswith(".py"):
                        module_name = module.split(".")[0]
                        module_path = os.path.join(tasks_group_directory, module)
                        module_task = cls.import_task(module_name, module_path)
                        setattr(tasks_group, module_name, module_task)
                setattr(tasks_factory, directory, tasks_group)

        return tasks_factory

    @staticmethod
    def import_task(name, path):
        """
        Import module task

        """
        spec = importlib.util.spec_from_file_location(name, path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return foo.Task(name)


if __name__ == '__main__':
    from logbook import StreamHandler, NestedSetup
    import sys

    # subscribe to instruments
    format_string = (
        '[{record.time:%Y-%m-%d %H:%M:%S.%f%z}] '
        '{record.level_name: <5} - {record.channel: <15}: {record.message}'
    )

    handlers = [
        StreamHandler(sys.stdout, level='INFO', format_string=format_string, bubble=True),
    ]

    with NestedSetup(handlers):
        tasks = TasksFactory.create()
        tasks.aws.start_instances.run("i-0ee55a73dbe79d0db", region_name="us-east-1b")
