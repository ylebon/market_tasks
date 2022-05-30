import boto3

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Create ec2 instance

    """

    def run(self, **kwargs):
        """
        Create amazon instance

        """
        # services config
        services_config = ConfigServices.create()

        # boto client
        ec2 = boto3.client(
            'ec2',
            aws_access_key_id=services_config.get_value("AMAZON.ACCESS_KEY"),
            aws_secret_access_key=services_config.get_value("AMAZON.SECRET_KEY"),
            region_name=kwargs.get("region_name", "us-west-1")
        )

        # kwargs min and max count
        kwargs['MinCount'] = kwargs.get('MinCount', 1)
        kwargs['MaxCount'] = kwargs.get('MaxCount', 1)

        instance = ec2.run_instances(
            **kwargs
        )
        print(instance)

