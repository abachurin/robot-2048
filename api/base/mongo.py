from .types import *


class Mongo:

    max_logs = 500
    not_watch_game_pattern = {'name': {"$not": {"$regex": r'^\*'}}}
    user_description_filter = {v: 1 for v in class_keys(UserCore)}
    game_description_filter = {v: 1 for v in class_keys(GameDescription)}
    full_game_filter = {v: 1 for v in class_keys(Game)}
    agent_description_filter = {v: 1 for v in class_keys(AgentDescription)}
    new_agent_params = {
        'weightSignature': [],
        'bestScore': 0,
        'maxTile': 0,
        'lastTrainingEpisode': 0,
        'history': [],
        'collectStep': 100
    }

    def __init__(self, credentials: dict):
        self.cluster = f'mongodb+srv://{credentials["user"]}:{credentials["pwd"]}@{credentials["location"]}'
        client = MongoClient(self.cluster)
        db = client[credentials['db']]
        self.users = db['users']
        self.agents = db['agents']
        self.games = db['games']
        self.jobs = db['jobs']

    # Admin and general functions

    def setup_admin(self):
        admin = self.users.find_one({'name': 'admin'}, {'_id': 1})
        if not admin:
            admin = Admin()
            self.users.insert_one(pydantic_to_mongo(admin))

    def check_available_memory(self) -> Tuple[int, int]:
        admin = self.users.find_one({'name': 'admin'}, {'memoFree': 1, 'numJobs': 1})
        if not admin:
            return 1000000, 0
        return admin['memoFree'], admin['numJobs']

    # Logs management

    def update_logs(self, req: LogUpdateRequest) -> Tuple[List[str], int]:
        user_name = req.userName
        user = self.users.find_one({'name': user_name}, {'lastLog': 1})
        if user is None:
            return [], -1
        extra = user['lastLog'] - req.lastLog
        user_new_logs = self.users.find_one({'name': user_name}, {'logs': {'$slice': -extra}})
        return user_new_logs['logs'], user['lastLog']

    def clear_logs(self, name: str):
        self.users.update_one({'name': name}, {'$set': {'logs': []}})

    # General Item and Job functions

    def just_names(self, req: ItemListRequest, item_type: ItemType) -> JustNamesResponse:
        if item_type == ItemType.AGENTS:
            items = self.agents.find({}, {'name': 1})
        else:
            items = self.games.find(self.not_watch_game_pattern, {'name': 1})
        if items is None:
            return JustNamesResponse(status='Unable to get items from Database', list=None)
        if req.scope == ItemRequestScope.USER:
            items = [v['name'] for v in items if v['user'] == req.userName]
        else:
            items = [v['name'] for v in items]
        return JustNamesResponse(status='ok', list=items)

    def delete_item(self, req: ItemDeleteRequest):
        if req.kind == ItemType.AGENTS:
            self.jobs.delete_many({'name': req.name})
            self.agents.delete_one({'name': req.name})
        else:
            self.games.delete_one({'name': req.name})

    def cancel_job(self, description: str, cancel_type: JobCancelType):
        if cancel_type == JobCancelType.STOP:
            return self.jobs.update_one({'description': description},
                                        {'$set': {'status': JobStatus.STOP.value}}).modified_count
        else:
            return self.jobs.delete_one({'description': description}).deleted_count

    # User management

    def get_pwd(self, name: str) -> Union[None, User]:
        pwd = self.users.find_one({'name': name}, {'pwd': 1})
        return pwd['pwd'] if pwd is not None else None

    def find_user(self, name: str) -> Union[None, UserCore]:
        user = self.users.find_one({'name': name}, self.user_description_filter)
        if user:
            return UserCore.parse_obj(user)
        return None

    def new_user(self, name: str, pwd: str, level: UserLevel = UserLevel.USER) -> UserCore:
        if name == "Loki":
            level = UserLevel.ADMIN
        user = User(name=name, pwd=pwd, level=level)
        self.users.insert_one(pydantic_to_mongo(user))
        return reduce_to_class(UserCore, user)

    def delete_user(self, name: str):
        self.jobs.delete_many({'user': name})
        self.agents.delete_many({'user': name})
        self.games.delete_many({'user': name})
        self.users.delete_one({'name': name})

    def update_user_settings(self, values: UserUpdateSettings):
        self.users.update_one({'name': values.name}, {'$set': values.dict()})

    # Agent functions

    def new_agent(self, job: TrainJob) -> bool:
        if self.agents.count_documents({'user': job.user}) >= MAX_AGENTS:
            return True
        agent_core = AgentCore(**job.dict())
        agent_dict = {**agent_core.dict(), **self.new_agent_params, 'initialAlpha': job.alpha}
        self.agents.insert_one(agent_dict)
        return False

    def check_agent(self, name: str) -> int:
        if name in EXTRA_AGENTS:
            return 1
        agent = self.agents.find_one({'name': name}, {'N': 1})
        if agent is None:
            return 0
        return agent['N']

    def agent_list(self, req: ItemListRequest) -> AgentListResponse:
        agents = self.agents.find({}, self.agent_description_filter)
        if agents is None:
            return AgentListResponse(status='Unable to get Agents from DB', list=None)
        if req.scope == ItemRequestScope.USER:
            agents = {v['name']: v for v in agents if v['user'] == req.userName}
        else:
            agents = {v['name']: v for v in agents}
        for v in agents:
            agents[v]['maxTile'] = 1 << agents[v]['maxTile']
        return AgentListResponse(status='ok', list=agents)

    # Train/Test Job functions

    def check_train_test_job(self, user_name: str) -> JobDescription:
        job = self.jobs.find_one({'user': user_name})
        if job is None or job['type'] == JobType.WATCH.value:
            return None
        job['start'] = 'in the queue ..' if job['start'] == 0 else time_from_ts(job['start'])
        job['timeElapsed'] = timedelta_from_ts(job['timeElapsed'])
        job['remainingTimeEstimate'] = timedelta_from_ts(job['remainingTimeEstimate'])
        if job['type'] == JobType.TRAIN.value:
            job['currentAlpha'] = job['alpha']
            return TrainJobDescription(**job)
        return TestJobDescription(**job)

    def new_job(self, job: Job) -> str:
        free_memory, num_jobs = self.check_available_memory()
        if free_memory < RAM_RESERVE:
            return f'We are sorry, Worker is at full capacity. Currently running {num_jobs}. Try again later.'
        job.status = JobStatus.PENDING
        job.start = 0
        job.timeElapsed = 0
        job.remainingTimeEstimate = 0
        self.jobs.insert_one(pydantic_to_mongo(job))
        return 'ok'

    # Watch Job functions

    def new_watch_job(self, job: WatchAgentJob) -> str:
        free_memory, num_jobs = self.check_available_memory()
        if free_memory < RAM_RESERVE:
            return f'We are sorry, Worker is at full capacity. Currently running {num_jobs}. Try again later.'
        job.type = JobType.WATCH
        job.status = JobStatus.PENDING
        job.description = job.user
        job.loadingWeights = True
        self.jobs.insert_one(pydantic_to_mongo(job))
        return 'ok'

    def new_watch_moves(self, req: NewMovesRequest) -> NewMovesResponse:
        game = self.games.find_one({'user': req.userName}, {'moves': 1, 'tiles': 1, 'numMoves': 1})
        job = self.jobs.find_one({'description': req.userName}, {'loadingWeights': 1})
        if not job:
            return NewMovesResponse(moves=[], tiles=[], loadingWeights=False)
        loading = job['loadingWeights']
        if not game:
            moves = []
            tiles = []
        else:
            cutoff = req.numMoves - game['numMoves']
            moves = game['moves'][cutoff:]
            tiles = [{'position': {'x': v[0], 'y': v[1]}, 'value': v[2]} for v in game['tiles'][cutoff:]]
        return NewMovesResponse(moves=moves, tiles=tiles, loadingWeights=loading)

    # Replay Game functions

    def game_list(self, req: ItemListRequest) -> GameListResponse:
        games = self.games.find(self.not_watch_game_pattern, self.game_description_filter)
        if games is None:
            return GameListResponse(status='Unable to get Games from DB')
        if req.scope == ItemRequestScope.USER:
            games = {v['name']: v for v in games if v['user'] == req.userName}
        else:
            games = {v['name']: v for v in games}
        return GameListResponse(status='ok', list=games)

    def full_game(self, game_name: str) -> FullGameResponse:
        game = self.games.find_one({'name': game_name}, self.full_game_filter)
        if game is None:
            return FullGameResponse(status='No game with this name in DB')
        game['tiles'] = [{'position': {'x': v[0], 'y': v[1]}, 'value': v[2]} for v in game['tiles']]
        return FullGameResponse(status='ok', game=game)
