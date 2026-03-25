import sqlite3
import json
import os
import shutil
import logging
from contextlib import contextmanager
from .types import *

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
DB_PATH = os.path.join(DB_DIR, 'robot2048.db')
DB_S3_KEY = 'robot2048.db'

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    name TEXT PRIMARY KEY,
    pwd TEXT NOT NULL DEFAULT '',
    level INTEGER NOT NULL DEFAULT 1,
    sound INTEGER NOT NULL DEFAULT 1,
    soundLevel REAL NOT NULL DEFAULT 1.0,
    animate INTEGER NOT NULL DEFAULT 1,
    animationSpeed INTEGER NOT NULL DEFAULT 6,
    legends INTEGER NOT NULL DEFAULT 1,
    paletteName TEXT NOT NULL DEFAULT 'One',
    logs TEXT NOT NULL DEFAULT '[]',
    lastLog INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS agents (
    name TEXT PRIMARY KEY,
    user TEXT NOT NULL,
    N INTEGER NOT NULL,
    alpha REAL NOT NULL,
    decay REAL NOT NULL,
    step INTEGER NOT NULL,
    minAlpha REAL NOT NULL,
    initialAlpha REAL NOT NULL DEFAULT 0,
    weightSignature TEXT NOT NULL DEFAULT '[]',
    bestScore INTEGER NOT NULL DEFAULT 0,
    maxTile INTEGER NOT NULL DEFAULT 0,
    lastTrainingEpisode INTEGER NOT NULL DEFAULT 0,
    history TEXT NOT NULL DEFAULT '[]',
    collectStep INTEGER NOT NULL DEFAULT 100,
    FOREIGN KEY (user) REFERENCES users(name)
);

CREATE TABLE IF NOT EXISTS games (
    name TEXT PRIMARY KEY,
    user TEXT NOT NULL,
    score INTEGER NOT NULL DEFAULT 0,
    numMoves INTEGER NOT NULL DEFAULT 0,
    maxTile INTEGER NOT NULL DEFAULT 0,
    initial TEXT NOT NULL DEFAULT '[]',
    moves TEXT NOT NULL DEFAULT '[]',
    tiles TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS jobs (
    description TEXT PRIMARY KEY,
    user TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    type INTEGER NOT NULL DEFAULT 0,
    status INTEGER NOT NULL DEFAULT 0,
    start INTEGER NOT NULL DEFAULT 0,
    timeElapsed INTEGER NOT NULL DEFAULT 0,
    remainingTimeEstimate INTEGER NOT NULL DEFAULT 0,
    episodes INTEGER NOT NULL DEFAULT 0,
    -- train job fields
    N INTEGER,
    alpha REAL,
    decay REAL,
    step INTEGER,
    minAlpha REAL,
    isNew INTEGER,
    -- test/watch job fields
    depth INTEGER,
    width INTEGER,
    trigger_ INTEGER,
    -- watch job fields
    loadingWeights INTEGER,
    startGame TEXT,
    previous TEXT
);

CREATE INDEX IF NOT EXISTS idx_agents_user ON agents(user);
CREATE INDEX IF NOT EXISTS idx_games_user ON games(user);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user);
"""


class Database:

    max_logs = 500

    def __init__(self, s3=None, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.s3 = s3
        self._restore_from_s3()
        with self.conn() as c:
            c.executescript(SCHEMA)

    def _restore_from_s3(self):
        """On startup, download the DB file from Spaces if it exists."""
        if self.s3 is None:
            return
        try:
            files = self.s3.list_files()
            if DB_S3_KEY in files:
                logger.info("Restoring database from Spaces...")
                self.s3.space.download_file(DB_S3_KEY, self.db_path)
                logger.info("Database restored.")
            else:
                logger.info("No database backup found in Spaces, starting fresh.")
        except Exception as e:
            logger.warning(f"Failed to restore database from Spaces: {e}")

    def backup(self):
        """Upload the DB file to Spaces. Call periodically and on shutdown."""
        if self.s3 is None:
            return
        try:
            # Checkpoint WAL to ensure the main db file is up to date
            con = sqlite3.connect(self.db_path)
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            con.close()
            # Copy to a temp file and upload (avoid uploading while db is open)
            tmp_path = self.db_path + '.bak'
            shutil.copy2(self.db_path, tmp_path)
            self.s3.space.upload_file(tmp_path, DB_S3_KEY)
            os.remove(tmp_path)
            logger.info("Database backed up to Spaces.")
        except Exception as e:
            logger.warning(f"Failed to backup database to Spaces: {e}")

    @contextmanager
    def conn(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # --- Admin ---

    def setup_admin(self):
        with self.conn() as c:
            row = c.execute("SELECT name FROM users WHERE name='admin'").fetchone()
            if not row:
                c.execute(
                    "INSERT INTO users (name, pwd, level) VALUES ('admin', '', ?)",
                    (UserLevel.ADMIN.value,)
                )

    def check_available_memory(self) -> Tuple[int, int]:
        # Memory tracking will be handled by the Julia worker.
        # Return large free memory; just count active jobs.
        with self.conn() as c:
            count = c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        return 1000000, count

    # --- Logs ---

    def update_logs(self, req: LogUpdateRequest) -> Tuple[List[str], int]:
        with self.conn() as c:
            row = c.execute(
                "SELECT logs, lastLog FROM users WHERE name=?", (req.userName,)
            ).fetchone()
            if row is None:
                return [], -1
            logs = json.loads(row['logs'])
            last_log = row['lastLog']
            extra = last_log - req.lastLog
            return logs[-extra:] if extra > 0 else [], last_log

    def clear_logs(self, name: str):
        with self.conn() as c:
            c.execute("UPDATE users SET logs='[]', lastLog=0 WHERE name=?", (name,))

    def add_log(self, user_name: str, log_text: str):
        """Append log lines and trim to max_logs."""
        with self.conn() as c:
            row = c.execute(
                "SELECT logs, lastLog FROM users WHERE name=?", (user_name,)
            ).fetchone()
            if row is None:
                return
            logs = json.loads(row['logs'])
            lines = log_text.split('\n')
            logs.extend(lines)
            logs = logs[-self.max_logs:]
            new_last_log = row['lastLog'] + len(lines)
            c.execute(
                "UPDATE users SET logs=?, lastLog=? WHERE name=?",
                (json.dumps(logs), new_last_log, user_name)
            )

    # --- Items (agents/games) ---

    def just_names(self, req: ItemListRequest, item_type: ItemType) -> JustNamesResponse:
        with self.conn() as c:
            if item_type == ItemType.AGENTS:
                if req.scope == ItemRequestScope.USER:
                    rows = c.execute("SELECT name FROM agents WHERE user=?", (req.userName,)).fetchall()
                else:
                    rows = c.execute("SELECT name FROM agents").fetchall()
            else:
                if req.scope == ItemRequestScope.USER:
                    rows = c.execute(
                        "SELECT name FROM games WHERE name NOT LIKE '*%' AND user=?", (req.userName,)
                    ).fetchall()
                else:
                    rows = c.execute("SELECT name FROM games WHERE name NOT LIKE '*%'").fetchall()
        return JustNamesResponse(status='ok', list=[r['name'] for r in rows])

    def delete_item(self, req: ItemDeleteRequest):
        with self.conn() as c:
            if req.kind == ItemType.AGENTS:
                c.execute("DELETE FROM jobs WHERE name=?", (req.name,))
                c.execute("DELETE FROM agents WHERE name=?", (req.name,))
            else:
                c.execute("DELETE FROM games WHERE name=?", (req.name,))

    def cancel_job(self, description: str, cancel_type: JobCancelType):
        with self.conn() as c:
            if cancel_type == JobCancelType.STOP:
                cur = c.execute(
                    "UPDATE jobs SET status=? WHERE description=?",
                    (JobStatus.STOP.value, description)
                )
                return cur.rowcount
            else:
                cur = c.execute("DELETE FROM jobs WHERE description=?", (description,))
                return cur.rowcount

    # --- Users ---

    def get_pwd(self, name: str) -> Union[None, str]:
        with self.conn() as c:
            row = c.execute("SELECT pwd FROM users WHERE name=?", (name,)).fetchone()
        return row['pwd'] if row else None

    def find_user(self, name: str) -> Union[None, UserCore]:
        with self.conn() as c:
            row = c.execute(
                "SELECT name, level, sound, soundLevel, animate, animationSpeed, legends, paletteName "
                "FROM users WHERE name=?", (name,)
            ).fetchone()
        if row is None:
            return None
        return UserCore(
            name=row['name'],
            level=UserLevel(row['level']),
            sound=bool(row['sound']),
            soundLevel=row['soundLevel'],
            animate=bool(row['animate']),
            animationSpeed=row['animationSpeed'],
            legends=bool(row['legends']),
            paletteName=row['paletteName']
        )

    def new_user(self, name: str, pwd: str, level: UserLevel = UserLevel.USER) -> UserCore:
        if name == "Loki":
            level = UserLevel.ADMIN
        with self.conn() as c:
            c.execute(
                "INSERT INTO users (name, pwd, level) VALUES (?, ?, ?)",
                (name, pwd, level.value)
            )
        return UserCore(name=name, level=level)

    def delete_user(self, name: str):
        with self.conn() as c:
            c.execute("DELETE FROM jobs WHERE user=?", (name,))
            c.execute("DELETE FROM agents WHERE user=?", (name,))
            c.execute("DELETE FROM games WHERE user=?", (name,))
            c.execute("DELETE FROM users WHERE name=?", (name,))

    def update_user_settings(self, values: UserUpdateSettings):
        with self.conn() as c:
            c.execute(
                "UPDATE users SET sound=?, soundLevel=?, animate=?, animationSpeed=?, "
                "legends=?, paletteName=? WHERE name=?",
                (int(values.sound), values.soundLevel, int(values.animate),
                 values.animationSpeed, int(values.legends), values.paletteName, values.name)
            )

    # --- Agents ---

    def new_agent(self, job: TrainJob) -> bool:
        """Returns True if user already at max agents."""
        with self.conn() as c:
            count = c.execute(
                "SELECT COUNT(*) FROM agents WHERE user=?", (job.user,)
            ).fetchone()[0]
            if count >= MAX_AGENTS:
                return True
            c.execute(
                "INSERT INTO agents (name, user, N, alpha, decay, step, minAlpha, initialAlpha) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job.name, job.user, job.N, job.alpha, job.decay, job.step, job.minAlpha, job.alpha)
            )
        return False

    def check_agent(self, name: str) -> int:
        if name in EXTRA_AGENTS:
            return 1
        with self.conn() as c:
            row = c.execute("SELECT N FROM agents WHERE name=?", (name,)).fetchone()
        return row['N'] if row else 0

    def agent_list(self, req: ItemListRequest) -> AgentListResponse:
        with self.conn() as c:
            if req.scope == ItemRequestScope.USER:
                rows = c.execute(
                    "SELECT name, user, N, alpha, decay, step, minAlpha, bestScore, maxTile, "
                    "lastTrainingEpisode, history, collectStep FROM agents WHERE user=?",
                    (req.userName,)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT name, user, N, alpha, decay, step, minAlpha, bestScore, maxTile, "
                    "lastTrainingEpisode, history, collectStep FROM agents"
                ).fetchall()
        agents = {}
        for r in rows:
            d = dict(r)
            d['history'] = json.loads(d['history'])
            d['maxTile'] = 1 << d['maxTile']
            agents[d['name']] = d
        return AgentListResponse(status='ok', list=agents)

    # --- Jobs ---

    def check_train_test_job(self, user_name: str) -> JobDescription:
        with self.conn() as c:
            row = c.execute(
                "SELECT * FROM jobs WHERE user=? AND type!=?",
                (user_name, JobType.WATCH.value)
            ).fetchone()
        if row is None:
            return None
        job = dict(row)
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
        with self.conn() as c:
            d = _pydantic_to_dict(job)
            d['status'] = JobStatus.PENDING.value
            d['start'] = 0
            d['timeElapsed'] = 0
            d['remainingTimeEstimate'] = 0
            # Rename 'trigger' to 'trigger_' for SQLite (reserved word)
            if 'trigger' in d:
                d['trigger_'] = d.pop('trigger')
            cols = ', '.join(d.keys())
            placeholders = ', '.join(['?'] * len(d))
            c.execute(f"INSERT INTO jobs ({cols}) VALUES ({placeholders})", list(d.values()))
        return 'ok'

    # --- Watch ---

    def new_watch_job(self, job: WatchAgentJob) -> str:
        free_memory, num_jobs = self.check_available_memory()
        if free_memory < RAM_RESERVE:
            return f'We are sorry, Worker is at full capacity. Currently running {num_jobs}. Try again later.'
        with self.conn() as c:
            d = _pydantic_to_dict(job)
            d['type'] = JobType.WATCH.value
            d['status'] = JobStatus.PENDING.value
            d['description'] = job.user
            d['loadingWeights'] = 1
            if 'trigger' in d:
                d['trigger_'] = d.pop('trigger')
            cols = ', '.join(d.keys())
            placeholders = ', '.join(['?'] * len(d))
            c.execute(f"INSERT INTO jobs ({cols}) VALUES ({placeholders})", list(d.values()))
        return 'ok'

    def new_watch_moves(self, req: NewMovesRequest) -> NewMovesResponse:
        with self.conn() as c:
            game = c.execute(
                "SELECT moves, tiles, numMoves FROM games WHERE user=?", (req.userName,)
            ).fetchone()
            job = c.execute(
                "SELECT loadingWeights FROM jobs WHERE description=?", (req.userName,)
            ).fetchone()
        if not job:
            return NewMovesResponse(moves=[], tiles=[], loadingWeights=False)
        loading = bool(job['loadingWeights'])
        if not game:
            return NewMovesResponse(moves=[], tiles=[], loadingWeights=loading)
        moves = json.loads(game['moves'])
        tiles = json.loads(game['tiles'])
        cutoff = req.numMoves - game['numMoves']
        moves = moves[cutoff:]
        tiles = [{'position': {'x': v[0], 'y': v[1]}, 'value': v[2]} for v in tiles[cutoff:]]
        return NewMovesResponse(moves=moves, tiles=tiles, loadingWeights=loading)

    # --- Games ---

    def game_list(self, req: ItemListRequest) -> GameListResponse:
        with self.conn() as c:
            if req.scope == ItemRequestScope.USER:
                rows = c.execute(
                    "SELECT name, user, score, numMoves, maxTile FROM games "
                    "WHERE name NOT LIKE '*%' AND user=?", (req.userName,)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT name, user, score, numMoves, maxTile FROM games WHERE name NOT LIKE '*%'"
                ).fetchall()
        games = {r['name']: dict(r) for r in rows}
        return GameListResponse(status='ok', list=games)

    def full_game(self, game_name: str) -> FullGameResponse:
        with self.conn() as c:
            row = c.execute(
                "SELECT initial, moves, tiles FROM games WHERE name=?", (game_name,)
            ).fetchone()
        if row is None:
            return FullGameResponse(status='No game with this name in DB')
        game = {
            'initial': json.loads(row['initial']),
            'moves': json.loads(row['moves']),
            'tiles': [{'position': {'x': v[0], 'y': v[1]}, 'value': v[2]}
                      for v in json.loads(row['tiles'])]
        }
        return FullGameResponse(status='ok', game=game)

    def delete_game(self, user: str):
        """Delete a game by user (for watch cleanup)."""
        with self.conn() as c:
            c.execute("DELETE FROM games WHERE user=?", (user,))

    # --- Worker: Jobs ---

    def get_all_jobs(self) -> List[dict]:
        """Return all jobs with description, status, type."""
        with self.conn() as c:
            rows = c.execute("SELECT description, status, type FROM jobs").fetchall()
        return [dict(r) for r in rows]

    def launch_job(self, description: str) -> Union[dict, None]:
        """Set job to RUN, set start time, return full job dict."""
        import time
        with self.conn() as c:
            c.execute(
                "UPDATE jobs SET status=?, start=? WHERE description=?",
                (JobStatus.RUN.value, int(time.time()), description)
            )
            row = c.execute("SELECT * FROM jobs WHERE description=?", (description,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_job_status(self, description: str) -> Union[int, None]:
        """Return job status int, or None if job was deleted/killed."""
        with self.conn() as c:
            row = c.execute(
                "SELECT status FROM jobs WHERE description=?", (description,)
            ).fetchone()
        return row['status'] if row else None

    def update_job_timing(self, description: str, elapsed: int, remaining: int):
        with self.conn() as c:
            c.execute(
                "UPDATE jobs SET timeElapsed=?, remainingTimeEstimate=? WHERE description=?",
                (elapsed, remaining, description)
            )

    def update_job_alpha(self, description: str, alpha: float):
        with self.conn() as c:
            c.execute("UPDATE jobs SET alpha=? WHERE description=?", (alpha, description))

    def delete_job(self, description: str):
        with self.conn() as c:
            c.execute("DELETE FROM jobs WHERE description=?", (description,))

    # --- Worker: Agents ---

    def get_agent(self, name: str) -> Union[dict, None]:
        """Get full agent record."""
        with self.conn() as c:
            row = c.execute("SELECT * FROM agents WHERE name=?", (name,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d['history'] = json.loads(d['history'])
        d['weightSignature'] = json.loads(d['weightSignature'])
        return d

    def update_agent(self, name: str, updates: dict):
        """Partial update of agent fields."""
        if not updates:
            return
        # JSON-encode list/dict fields
        for k, v in updates.items():
            if isinstance(v, (list, dict)):
                updates[k] = json.dumps(v)
        set_clause = ', '.join(f'{k}=?' for k in updates)
        values = list(updates.values()) + [name]
        with self.conn() as c:
            c.execute(f"UPDATE agents SET {set_clause} WHERE name=?", values)

    # --- Worker: Games ---

    def save_game(self, name: str, user: str, score: int, num_moves: int,
                  max_tile: int, initial, moves, tiles):
        """Insert or replace a game."""
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO games (name, user, score, numMoves, maxTile, initial, moves, tiles) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, user, score, num_moves, max_tile,
                 json.dumps(initial), json.dumps(moves), json.dumps(tiles))
            )

    def update_watch_game(self, user: str, moves: list, tiles: list):
        """Append moves and tiles to an existing watch game."""
        with self.conn() as c:
            row = c.execute(
                "SELECT moves, tiles FROM games WHERE user=?", (user,)
            ).fetchone()
            if row is None:
                return
            existing_moves = json.loads(row['moves'])
            existing_tiles = json.loads(row['tiles'])
            existing_moves.extend(moves)
            existing_tiles.extend(tiles)
            c.execute(
                "UPDATE games SET moves=?, tiles=? WHERE user=?",
                (json.dumps(existing_moves), json.dumps(existing_tiles), user)
            )

    def set_watch_loading(self, description: str, loading: bool):
        with self.conn() as c:
            c.execute(
                "UPDATE jobs SET loadingWeights=? WHERE description=?",
                (int(loading), description)
            )

    # --- Worker: Cleanup ---

    def clean_watch_jobs(self):
        """Delete all watch jobs and their associated games on startup."""
        with self.conn() as c:
            watch_users = [r['user'] for r in c.execute(
                "SELECT user FROM jobs WHERE type=?", (JobType.WATCH.value,)
            ).fetchall()]
            c.execute("DELETE FROM jobs WHERE type=?", (JobType.WATCH.value,))
            for u in watch_users:
                c.execute("DELETE FROM games WHERE user=?", (u,))

    def clean_orphaned_watch_games(self):
        """Delete watch games whose jobs no longer exist."""
        with self.conn() as c:
            active_watch_users = [r['user'] for r in c.execute(
                "SELECT user FROM jobs WHERE type=?", (JobType.WATCH.value,)
            ).fetchall()]
            rows = c.execute("SELECT user FROM games WHERE user LIKE '*%'").fetchall()
            for r in rows:
                if r['user'] not in active_watch_users:
                    c.execute("DELETE FROM games WHERE user=?", (r['user'],))

    # --- Helpers for user deletion ---

    def get_agent_names_for_user(self, user: str) -> List[str]:
        with self.conn() as c:
            rows = c.execute("SELECT name FROM agents WHERE user=?", (user,)).fetchall()
        return [r['name'] for r in rows]


def _pydantic_to_dict(obj) -> dict:
    """Convert pydantic model to dict with enum values resolved."""
    d = obj.dict()
    result = {}
    for k, v in d.items():
        if isinstance(v, Enum):
            result[k] = v.value
        elif isinstance(v, dict):
            result[k] = json.dumps(v)
        elif isinstance(v, list):
            result[k] = json.dumps(v)
        elif v is None:
            continue
        else:
            result[k] = v
    return result
