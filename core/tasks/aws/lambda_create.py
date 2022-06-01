import boto3

from core.task_step import TaskStep


class Task(TaskStep):
    """
    Deploy lambda function

    """

    def run(self, function_name, handler, timeout=300, python_runtime='python3.6'):
        """Deploy lambda function"""

        lambda_client = boto3.client('lambda')
        env_variables = dict()

        with open('lambda.zip', 'rb') as f:
            zipped_code = f.read()

        lambda_client.create_function(
            FunctionName=function_name,
            Runtime=python_runtime,
            Handler=handler,
            Code=dict(ZipFile=zipped_code),
            Timeout=timeout,
            Environment=dict(Variables=env_variables),
        )
