# API Component

FastAPI server handling user management, job orchestration, and data persistence.

## Location

`api/`

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, all user-facing endpoints |
| `worker_routes.py` | `/worker/*` endpoints for Julia worker |
| `base/start.py` | Imports, constants, helper functions |
| `base/types.py` | Pydantic models, enums |
| `base/database.py` | SQLite `Database` class with all DB operations |
| `base/storage.py` | S3/Spaces wrapper (`Storage` class) |
| `base/utils.py` | Glue: creates `DB` and `S3` instances, delete helpers |
| `Dockerfile` | Python 3.11 slim, pip install, runs `main.py` |

## Database (SQLite)

Four tables:

- **users**: accounts, settings, logs (JSON array)
- **agents**: RL agent metadata (N, alpha, decay, history, best score)
- **jobs**: pending/running/stopped jobs (train, test, watch)
- **games**: saved game replays (moves, tiles, score)

DB file: `api/data/robot2048.db` (gitignored). Backed up to S3 periodically when deployed.

## User-Facing Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/users/login` | Login |
| POST | `/users/register` | Register |
| DELETE | `/users/delete` | Delete account |
| PUT | `/users/settings` | Update preferences |
| POST | `/logs/update` | Poll for new log lines |
| PUT | `/logs/clear` | Clear logs |
| POST | `/jobs/train` | Start training job |
| POST | `/jobs/test` | Start test job |
| POST | `/jobs/cancel` | Stop/kill a job |
| POST | `/jobs/description` | Get current job status |
| POST | `/agents/list` | List agents |
| POST | `/agents/just_names` | List agent names only |
| POST | `/games/list` | List saved games |
| GET | `/games/{name}` | Full game replay data |
| DELETE | `/item/delete` | Delete agent or game |
| POST | `/watch/new_agent` | Start watch-agent-play job |
| POST | `/watch/new_moves` | Poll for new moves in watch game |
| DELETE | `/watch/cancel` | Cancel watch job |

## Worker Endpoints (`/worker/*`)

Internal endpoints called by the Julia worker:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/worker/jobs` | List all jobs with status |
| POST | `/worker/jobs/{desc}/launch` | Set job to RUN, return full job |
| GET | `/worker/jobs/{desc}/status` | Lightweight status check |
| PUT | `/worker/jobs/{desc}/timing` | Update elapsed/remaining time |
| PUT | `/worker/jobs/{desc}/alpha` | Update learning rate |
| DELETE | `/worker/jobs/{desc}` | Delete completed job |
| GET | `/worker/agents/{name}` | Get agent metadata |
| PUT | `/worker/agents/{name}` | Update agent metadata |
| POST | `/worker/agents/{name}/weights` | Upload weights binary |
| GET | `/worker/agents/{name}/weights` | Download weights binary |
| POST | `/worker/games` | Save a game |
| PUT | `/worker/games/{user}/moves` | Append moves to watch game |
| DELETE | `/worker/games/{user}` | Delete watch game |
| POST | `/worker/logs` | Add log entry |
| PUT | `/worker/watch/{desc}/loading` | Set watch loading flag |
| POST | `/worker/cleanup` | Clean stale watch jobs on startup |

## Job Lifecycle

```
User request → API creates job (PENDING)
                    ↓
Worker polls → launches job (RUN)
                    ↓
Worker executes → updates timing/logs/weights
                    ↓
Complete or STOP → Worker deletes job
```

Job statuses: `PENDING(0)`, `RUN(1)`, `STOP(2)`

## Constants

- `RAM_RESERVE = 500` MB — minimum free RAM to accept new job
- `MAX_AGENTS = 5` per user
- `MAX_CONCURRENT_JOBS = 5`
- `WORKER_TOTAL_RAM = 4000` MB
- `MAX_N_USER = 5`, `MAX_N_ADMIN = 6`
