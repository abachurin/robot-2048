from .start import *


# Helper functions
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def pydantic_to_mongo(obj) -> dict:
    obj_json = json.dumps(obj.dict(), cls=CustomJSONEncoder)
    return json.loads(obj_json)


def reduce_to_class(cls, obj):
    return cls(**obj.dict())


def class_keys(cls: type) -> List[str]:
    keys = []
    for c in reversed(cls.mro()):
        if hasattr(c, "__annotations__"):
            keys.extend(c.__annotations__.keys())
    return keys


# Enum

class UserLevel(Enum):
    GUEST = 0
    USER = 1
    ADMIN = 2


class JobType(Enum):
    TRAIN = 0
    TEST = 1
    WATCH = 2


class JobStatus(Enum):
    PENDING = 0
    RUN = 1
    STOP = 2


class JobCancelType(Enum):
    STOP = "STOP"
    KILL = "KILL"


class ItemRequestScope(Enum):
    USER = "user"
    ALL = "all"


class ItemType(Enum):
    AGENTS = "Agents"
    GAMES = "Games"


class Admin(BaseModel):
    name: str = "admin"
    memoUsed: int = 0
    memoFree: int = 0
    s3Used: int = 0
    mongoUsed: int = 0
    numJobs: int = 0
    logs: list[str] = []
    lastLog: int = 0


class UserName(BaseModel):
    userName: str


class ItemListRequest(UserName):
    scope: ItemRequestScope


class ItemDeleteRequest(BaseModel):
    name: str
    kind: ItemType


class JobCancelRequest(BaseModel):
    description: str
    type: JobCancelType


class JustNamesResponse(BaseModel):
    status: str
    list: Optional[List[str]]


# User management

class UserLogin(BaseModel):
    name: str
    pwd: str


class UserCore(BaseModel):
    name: str = ""
    level: UserLevel = UserLevel.USER
    sound: bool = True
    soundLevel: float = 1.0
    animate: bool = True
    animationSpeed: int = 6
    legends: bool = True
    paletteName: str = "One"


class User(UserCore):
    pwd: str = ""
    logs: list[str] = []
    lastLog: int = 0


class UserLoginResponse(BaseModel):
    status: str
    content: Optional[UserCore]


class UserUpdateSettings(BaseModel):
    name: str
    sound: bool
    soundLevel: float
    animate: bool
    animationSpeed: int
    legends: bool
    paletteName: str


# Agent

class AgentCore(BaseModel):
    user: str
    name: str
    N: int
    alpha: float
    decay: float
    step: int
    minAlpha: float


class AgentDescription(AgentCore):
    bestScore: int
    maxTile: int
    lastTrainingEpisode: int
    history: List[float]
    collectStep: int


class Agent(AgentDescription):
    weightSignature: List[int]
    initialAlpha: float


class AgentListResponse(BaseModel):
    status: str
    list: Optional[Dict[str, AgentDescription]]


# Train/Test Job

class BaseJob(BaseModel):
    description: Optional[str]
    type: Optional[JobType]
    status: Optional[JobStatus]
    start: Optional[int]
    timeElapsed: Optional[int]
    remainingTimeEstimate: Optional[int]


class TrainJob(AgentCore, BaseJob):
    episodes: int
    isNew: bool


class TestWatchJobBase(BaseJob):
    user: str
    name: str
    depth: int
    width: int
    trigger: int


class TestJob(TestWatchJobBase):
    episodes: int


Job: type = Union[TrainJob, TestJob, None]


# Job Description

class JobDescriptionBase(BaseModel):
    type: JobType
    description: str
    name: str
    episodes: int
    start: str
    timeElapsed: str
    remainingTimeEstimate: str


class TrainJobDescription(JobDescriptionBase):
    currentAlpha: float


class TestJobDescription(JobDescriptionBase):
    depth: int
    width: int
    trigger: int


JobDescription = Union[TrainJobDescription, TestJobDescription, None]


class JobUpdateResponse(BaseModel):
    status: str
    job: Optional[JobDescription]


# Logs

class LogUpdateRequest(UserName):
    lastLog: int


class LogsUpdateResponse(BaseModel):
    status: str
    logs: Optional[list[str]]
    lastLog: int


# Games

class Offset(BaseModel):
    x: int
    y: int


class GameTile(BaseModel):
    position: Offset
    value: int


class GameDescription(BaseModel):
    user: str
    name: str
    score: int
    numMoves: int
    maxTile: int


class Game(BaseModel):
    initial: List[List[int]]
    moves: List[int]
    tiles: List[GameTile]


class GameListResponse(BaseModel):
    status: str
    list: Optional[Dict[str, GameDescription]]


class FullGameResponse(BaseModel):
    status: str
    game: Optional[Game]


# Watch Job

class GameWatch(BaseModel):
    initial: List[List[int]]
    score: int
    numMoves: int


class WatchAgentJob(TestWatchJobBase):
    startGame: GameWatch
    previous: str
    loadingWeights: Optional[bool]


class GameWatchNew(BaseModel):
    user: str
    startGame: GameWatch


class NewMovesRequest(UserName):
    numMoves: int


class NewMovesResponse(BaseModel):
    moves: List[int]
    tiles: List[GameTile]
    loadingWeights: bool
