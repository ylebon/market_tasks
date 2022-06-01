import shlex
from datetime import datetime

import boto3

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Run fargate task

    """
    client = None

    def run(self, name, cluster, command, environment=None, group=None, count=1, memory=1024, cpu=512,
            task_def='varatra-task', launch_type='FARGATE',
            region="eu-west-1", tags=None):
        # services config
        services_config = ConfigServices.create()

        # command
        if isinstance(command, str):
            command = shlex.split(command)

        # boto client
        if not self.client:
            self.client = boto3.client(
                'ecs',
                aws_access_key_id=services_config.get_value("AMAZON.ACCESS_KEY"),
                aws_secret_access_key=services_config.get_value("AMAZON.SECRET_KEY"),
                region_name=region
            )

        # run task
        environment = environment or list()
        self.log.info(f"msg='running task' container='{name}' cluster='{cluster}'")

        group = group or datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
        response = self.client.run_task(
            cluster=cluster,
            count=count,
            group=group,
            launchType=launch_type,
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': [
                        'subnet-0f20fb347f11e7124',
                    ],
                    'securityGroups': [
                        'sg-01dd368c7a56b7f7a',
                    ],
                    'assignPublicIp': 'ENABLED'
                }
            },
            overrides={
                'containerOverrides': [
                    {
                        'name': 'varatra-container',
                        'command': command,
                        'environment': environment,
                        'cpu': cpu,
                        'memory': memory,
                    },
                ],
                'cpu': str(cpu),
                'memory': str(memory)
            },
            taskDefinition=task_def
        )

        return response


if __name__ == "__main__":
    from logbook import StreamHandler
    import sys

    StreamHandler(sys.stdout).push_application()
    t = Task('fargate_task')
    response = t.run('container-1', 'backtesting', '/bin/ls', task_def='varatra-task:6')
    print(response)
