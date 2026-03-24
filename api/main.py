from contextlib import asynccontextmanager
from base.utils import *

RESTRICTED_USERNAMES = {"Login", "login", "Loki", "admin", "Admin"}

BACKUP_INTERVAL = 300  # seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    DB.setup_admin()
    # Periodic backup task
    import asyncio

    async def periodic_backup():
        while True:
            await asyncio.sleep(BACKUP_INTERVAL)
            DB.backup()

    task = asyncio.create_task(periodic_backup())
    yield
    # Shutdown: cancel backup task, do a final backup
    task.cancel()
    DB.backup()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
async def root():
    return {'message': 'Robot-2048 backend, made with FastAPI, go to /docs to see all available endpoints'}


# User management

@app.post('/users/login')
async def users_login(login_data: UserLogin) -> UserLoginResponse:
    user_name = login_data.name
    pwd = login_data.pwd
    db_pwd = DB.get_pwd(user_name)
    if db_pwd is None:
        return UserLoginResponse(status=f'User {user_name} does not exist', content=None)
    if db_pwd != pwd:
        return UserLoginResponse(status=f'Wrong password for {user_name}!', content=None)
    db_user = DB.find_user(user_name)
    return UserLoginResponse(status='ok', content=db_user)


@app.post('/users/register')
async def users_register(login_data: UserLogin) -> UserLoginResponse:
    user_name = login_data.name
    pwd = login_data.pwd
    db_pwd = DB.get_pwd(user_name)
    if db_pwd is not None or user_name in RESTRICTED_USERNAMES:
        return UserLoginResponse(status=f'User {user_name} already exists!', content=None)
    new_user = DB.new_user(user_name, pwd)
    return UserLoginResponse(status='ok', content=new_user)


@app.delete('/users/delete')
async def users_delete(login_data: UserLogin) -> UserLoginResponse:
    user_name = login_data.name
    pwd = login_data.pwd
    db_pwd = DB.get_pwd(user_name)
    if db_pwd is None:
        return UserLoginResponse(status=f'User {user_name} does not exist', content=None)
    if db_pwd != pwd:
        return UserLoginResponse(status=f'Wrong password for {user_name}!', content=None)
    delete_user_total(user_name)
    return UserLoginResponse(status='ok', content=None)


@app.put('/users/settings')
async def users_settings(values: UserUpdateSettings) -> Response:
    user_name = values.name
    db_user = DB.find_user(user_name)
    if db_user is None:
        return Response(status_code=404)
    DB.update_user_settings(values)
    return Response(status_code=200)


# Logs and Job Description

@app.post('/logs/update')
async def logs_update(req: LogUpdateRequest) -> LogsUpdateResponse:
    user_name = req.userName
    new_logs, last_log = DB.update_logs(req)
    if last_log == -1 and user_name != "Login":
        return LogsUpdateResponse(status=f'User {user_name} does not exist')
    return LogsUpdateResponse(status="ok", logs=new_logs, lastLog=last_log)


@app.put('/logs/clear')
async def logs_clear(clear_request: UserName):
    user_name = clear_request.userName
    DB.clear_logs(user_name)


@app.post('/jobs/description')
async def jobs_description(job_request: UserName) -> JobUpdateResponse:
    user_name = job_request.userName
    job = DB.check_train_test_job(user_name)
    if job is None:
        return JobUpdateResponse(status=f"No running jobs for {user_name}")
    return JobUpdateResponse(status=f"ok", job=job)


# General Agent/Game/Job endpoints

@app.post('/agents/list')
async def agents_list(agent_list_request: ItemListRequest) -> AgentListResponse:
    return DB.agent_list(agent_list_request)


@app.post('/games/list')
async def games_list(game_list_request: ItemListRequest) -> GameListResponse:
    return DB.game_list(game_list_request)


@app.post('/agents/just_names')
async def just_agent_names(agent_list_request: ItemListRequest) -> JustNamesResponse:
    return DB.just_names(agent_list_request, ItemType.AGENTS)


@app.delete('/item/delete')
async def item_delete(delete_item_request: ItemDeleteRequest):
    delete_item_total(delete_item_request)


@app.get('/games/{game_name}')
async def full_game(game_name: str) -> FullGameResponse:
    return DB.full_game(game_name)


@app.post('/jobs/cancel')
async def jobs_cancel(cancel_request: JobCancelRequest) -> str:
    description = cancel_request.description
    cancel_type = cancel_request.type
    count = DB.cancel_job(description, cancel_type)
    if not count:
        return "Job doesn't exist anymore"
    return f'{cancel_type.value} "{description}": processing request'


# Train/Test endpoints

@app.post('/jobs/train')
async def jobs_train(job: TrainJob) -> str:
    running_now = DB.check_train_test_job(job.user)
    if running_now is not None:
        return f"Currently running {running_now}. Stop/Kill it first."
    agent_exists = DB.check_agent(job.name)
    if job.isNew:
        if agent_exists:
            return f"Agent {job.name} already exists, choose another name"
        if DB.new_agent(job):
            return f"You already have a maximum of {MAX_AGENTS} Agents, delete some before creating new!"
    elif not agent_exists:
        return f"Agent {job.name} doesn't exist anymore"
    if job.episodes:
        job.description = f"Training {job.name} for {job.user}"
        job.type = JobType.TRAIN
        return DB.new_job(job)


@app.post('/jobs/test')
async def jobs_test(job: TestJob) -> str:
    running_now = DB.check_train_test_job(job.user)
    if running_now is not None:
        return f"Currently running {running_now}. Stop/Kill it first."
    n = DB.check_agent(job.name)
    if not n:
        return f"Agent {job.name} doesn't exist"
    job.description = f"Testing {job.name} for {job.user}"
    job.type = JobType.TEST
    return DB.new_job(job)


# Watch Agent Job endpoints

@app.post('/watch/new_agent')
async def watch_new_agent(job: WatchAgentJob) -> str:
    previous_user = job.previous
    if previous_user:
        DB.cancel_job(previous_user, JobCancelType.KILL)
        DB.delete_game(previous_user)
    n = DB.check_agent(job.name)
    if not n:
        return f"Agent {job.name} doesn't exist"
    return DB.new_watch_job(job)


@app.post('/watch/new_moves')
async def watch_new_moves(req: NewMovesRequest) -> NewMovesResponse:
    return DB.new_watch_moves(req)


@app.delete('/watch/cancel')
async def watch_job_cancel(req: UserName):
    DB.cancel_job(req.userName, JobCancelType.KILL)
    DB.delete_game(req.userName)


if __name__ == '__main__':

    uvicorn.run(app, host="0.0.0.0", port=5000)
