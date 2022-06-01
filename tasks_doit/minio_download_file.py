from core.minio import download_file


def run(bucket_name, source, destination):
    task = download_file.Task("download_file")
    task.run(bucket_name, source, destination)
