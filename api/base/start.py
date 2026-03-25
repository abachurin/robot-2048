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
MAX_CONCURRENT_JOBS = 5
WORKER_TOTAL_RAM = 4000  # basic-m instance, MB
MAX_N_USER = 5
MAX_N_ADMIN = 6

# Estimated RAM per agent in MB, by N (includes Julia overhead per job)
AGENT_RAM_MB = {
    2: 1,
    3: 2,
    4: 10,
    5: 30,
    6: 900,   # worst case cutoff=15; cutoff=13 is ~400MB
}


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
