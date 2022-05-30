import boto3

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Create ec2 instance

    """

    def run(self, instance_ids, region_name="us-east-1b", **kwargs):
        """
        Create amazon instance

        """
        # create instances ids
        if isinstance(instance_ids, str):
            instance_ids = [instance_ids]

        # services config
        services_config = ConfigServices.create()

        # boto client
        ec2 = boto3.client(
            'ec2',
            aws_access_key_id=services_config.get_value("AMAZON.ACCESS_KEY"),
            aws_secret_access_key=services_config.get_value("AMAZON.SECRET_KEY"),
            region_name=region_name
        )

        response = ec2.start_instances(
            InstanceIds=instance_ids,
            **kwargs
        )

        print(response)

