from .start import *


class Storage:

    def __init__(self, credentials: dict):
        qwargs = {
            'service_name': 's3',
            'endpoint_url': f'https://{credentials["region"]}.digitaloceanspaces.com',
            'region_name': credentials['region'],
            'aws_access_key_id': credentials['access_key'],
            'aws_secret_access_key': credentials['secret_key']
        }
        self.engine = boto3.resource(**qwargs)
        self.client = boto3.client(**qwargs)
        self.space_name = credentials['space']
        self.space = self.engine.Bucket(self.space_name)

    def list_files(self):
        return [o.key for o in self.space.objects.all()]

    def delete(self, name):
        key = full_key(name)
        if key in self.list_files():
            self.engine.Object(self.space_name, key).delete()

    def upload(self, local_path: str, name: str):
        key = full_key(name)
        self.space.upload_file(local_path, key)

    def download(self, name: str, local_path: str) -> bool:
        key = full_key(name)
        if key not in self.list_files():
            return False
        self.space.download_file(key, local_path)
        return True
