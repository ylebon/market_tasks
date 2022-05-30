from varatra_tasks.core.tasks.minio import upload_file


def run(bucket_name, source, destination):
    task = upload_file.Task("upload_file")
    task.run(bucket_name, source, destination)
