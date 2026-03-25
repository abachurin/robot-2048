from .storage import *
from .database import *

working_directory = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(working_directory, 'config.json'), 'r') as f:
    CONF = json.load(f)
LOCAL = os.environ.get('AT_HOME', 'local') == 'local'

if LOCAL:
    with open(CONF['s3_credentials'], 'r') as f:
        s3_credentials = json.load(f)
else:
    s3_credentials = {
        'region': os.getenv('S3_REGION', None),
        'space': os.getenv('S3_SPACE', 'robot-2048'),
        'access_key': os.getenv('S3_ACCESS_KEY', None),
        'secret_key': os.getenv('S3_SECRET_KEY', None)
    }


S3 = Storage(s3_credentials)
DB = Database(s3=None if LOCAL else S3)


def delete_item_total(delete_item_request: ItemDeleteRequest):
    DB.delete_item(delete_item_request)
    if delete_item_request.kind == ItemType.AGENTS:
        S3.delete(delete_item_request.name)


def delete_user_total(name: str):
    for agent_name in DB.get_agent_names_for_user(name):
        S3.delete(agent_name)
    DB.delete_user(name)
