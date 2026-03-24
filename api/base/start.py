from datetime import datetime, timedelta
import os
import json
import re
from pprint import pprint
from typing import Union, Tuple, Optional, List, Dict
from pydantic import BaseModel
from enum import Enum
import boto3
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

RAM_RESERVE = 500
MAX_AGENTS = 5


def full_key(name):
    return f'{name}.pkl'


def time_suffix():
    return str(datetime.now())[-6:]


def temp_local():
    return f'tmp{time_suffix()}.pkl'


def time_now():
    return str(datetime.now())[:19]


def temp_watch_job():
    return 'watch' + ''.join([v for v in str(datetime.now()) if v.isdigit()])


def time_from_ts(ts: int):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def timedelta_from_ts(ts: int):
    return str(timedelta(seconds=ts))


EXTRA_AGENTS = ['Random Moves', 'Best Score']
