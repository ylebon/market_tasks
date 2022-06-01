from core.minio import create_bucket


def run(name):
    task = create_bucket.Task("create_bucket")
    task.run(name)
