import shlex

from libcloud.compute.providers import get_driver
from libcloud.compute.types import Provider

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Run fargate task

    """
    client = None

    def run(self, name, command, size='DEV1-S', image='Docker'):
        # services config
        services_config = ConfigServices.create()

        # command
        if isinstance(command, str):
            command = shlex.split(command)

        # cls driver
        cls = get_driver(Provider.SCALEWAY)
        self.client = cls(
            services_config.get_value('SCALEWAY.ACCESS_KEY'),
            services_config.get_value('SCALEWAY.SECRET_TOKEN')
        )

        sizes = self.client.list_sizes()
        images = self.client.list_images()

        size = [s for s in sizes if s.id == size][0]
        image = [i for i in images if image in i.name][1]

        image.extra['size'] = 0

        self.log.info(f"msg='running scaleway node' name='{name}' size='{size.id}' image='{image.name}'")

        node = self.client.create_node(name=name, size=size, image=image)
        print(node)


if __name__ == "__main__":
    from logbook import StreamHandler
    import sys

    StreamHandler(sys.stdout).push_application()
    t = Task('scaleway_task')
    response = t.run('test', 'ls', size='C2M')
    print(response)
