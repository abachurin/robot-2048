# Julia Worker Component

RL training/testing engine written in Julia. Polls the API for jobs and executes them.

## Location

`julia-worker/`

## Key Files

| File | Purpose |
|------|---------|
| `main.jl` | Entry point — polling loop, job dispatch |
| `src/Game2048.jl` | Module definition, includes all submodules |
| `src/board.jl` | Board as UInt64, cell ops, move execution |
| `src/lookup.jl` | Pre-computed row/column move tables (65K entries) |
| `src/symmetry.jl` | D4 symmetry group (8 transformations) |
| `src/features.jl` | N-tuple feature extraction (N=2..6) |
| `src/weights.jl` | WeightTable struct, evaluate/update, save/load |
| `src/game.jl` | GameState, new_game, lookahead search |
| `src/agent.jl` | QAgent, TD(0) training, test_run |
| `src/worker_client.jl` | HTTP client for API `/worker/*` endpoints |
| `src/job_runner.jl` | Job execution (train/test/watch) |
| `Dockerfile` | Julia 1.11, Pkg.instantiate, runs `main.jl` |

## Game2048 Module

### Board Representation

Single `UInt64` packs 4x4 grid into 16 cells x 4 bits each.
- Cell values: 0=empty, k=tile 2^k (1→2, 2→4, ..., 15→32768)
- Moves via pre-computed lookup tables for instant row/column operations

### N-Tuple Features

| N | Groups | Weights | RAM (Float32) |
|---|--------|---------|---------------|
| 2 | 24 | 6K | ~24 KB |
| 3 | 52 | 213K | ~0.8 MB |
| 4 | 17 | 1.1M | ~4 MB |
| 5 | 21 | 5.3M | ~20 MB |
| 6 (cutoff=13) | 33 | 95M | ~365 MB |
| 6 (cutoff=15) | 33 | 206M | ~790 MB |

N=6 features use a configurable cutoff (`N6_CUTOFF`) to clamp cell values, reducing weight table size.

### QAgent

TD(0) value learning with:
- gamma=1 (episodic, no discounting)
- epsilon=0 (greedy — game is stochastic enough)
- D4 symmetry: each update touches 8x more weights
- Learning rate modes: `GLOBAL_DECAY`, `CONSTANT`, `FRONTIER`

### Weight Persistence

Binary format: `[n, last_episode, best_score, max_tile, alpha, num_weights, weights...]`

Saved to/loaded from S3 via API endpoints.

## Worker Loop (`main.jl`)

```
startup → wait for API → cleanup stale watch jobs
    ↓
every 3s:
    GET /worker/jobs
    for each PENDING job:
        POST /worker/jobs/{desc}/launch
        spawn Julia Task → run_job()
    check running tasks for completion
```

Environment:
- `API_HOST`: when set (on DO), connects to that URL instead of localhost:8000
- Weights cached locally in `julia-worker/weights/`

## Job Types

### Train (`run_train_job`)
1. Create QAgent with job parameters
2. Download existing weights from S3 (if any)
3. Run training episodes
4. Every 100 episodes: log progress, update timing
5. Every 1000 episodes: save weights to S3, update agent metadata
6. Check job status each episode (stop if STOP/deleted)

### Test (`run_test_job`)
1. Load agent weights
2. Play games with lookahead search (depth/width/trigger)
3. Log each game result
4. Save best game to DB

### Watch (`run_watch_job`)
1. Load agent weights
2. Play game move-by-move
3. Every 2s: flush moves/tiles to DB via API
4. Frontend polls for new moves to animate
5. Game over: wait 10s, clean up

## Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `benchmark.jl` | Basic N-tuple training benchmark |
| `benchmark_float.jl` | Float16 vs Float32 vs Float64 comparison |
| `benchmark_strategies.jl` | Alpha mode comparison |
| `benchmark_transfer.jl` | Transfer learning evaluation |
| `run_constant025.jl` | Constant alpha=0.25 study |

## Dependencies

- `HTTP.jl` — API communication
- `JSON3.jl` — JSON parsing
- `Random` — RNG (stdlib)
