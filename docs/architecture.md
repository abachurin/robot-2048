# Robot-2048 Architecture

## Overview

Robot-2048 is a web application for training and testing reinforcement learning agents that play the 2048 game. Users can create agents with different N-tuple feature sizes, train them, test with lookahead search, and watch agents play in real-time.

## Stack

| Component | Technology | Location | Port |
|-----------|-----------|----------|------|
| UI | React + TypeScript + Vite | `ts-vite-ui/` | 5173 (dev) |
| API | Python + FastAPI + SQLite | `api/` | 8000 |
| Worker | Julia + HTTP.jl | `julia-worker/` | — (polls API) |
| Storage | DigitalOcean Spaces (S3) | — | — |

## Component Communication

```
┌──────────┐     HTTP      ┌──────────┐     HTTP      ┌──────────────┐
│          │ ──────────── > │          │ < ─────────── │              │
│    UI    │   /users/*     │   API    │   /worker/*   │ Julia Worker │
│  (React) │   /jobs/*      │ (FastAPI)│   (polling)   │  (Game2048)  │
│          │   /agents/*    │          │               │              │
└──────────┘                └────┬─────┘               └──────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
               ┌────▼───┐  ┌────▼───┐  ┌─────▼────┐
               │ SQLite │  │   S3   │  │ S3       │
               │  (DB)  │  │(backup)│  │(weights) │
               └────────┘  └────────┘  └──────────┘
```

## Data Flow

1. **User creates agent** → UI → API → SQLite (agent record)
2. **User starts training** → API creates job (status=PENDING) in SQLite
3. **Worker polls** `/worker/jobs` every 3s → finds pending job → launches it
4. **During training** → Worker calls API to update timing, logs, alpha
5. **Checkpoint** → Worker saves weights to S3 via API, updates agent metadata
6. **Job complete** → Worker deletes job from SQLite

## Deployment

- **Hosting**: DigitalOcean App Platform (FRA1 region)
- **CI/CD**: Push to `master` on `abachurin/robot-2048` → auto-deploy
- **Domain**: robot2048.com
- **Local dev**: Run `robot2048` command to start all components

## Resource Limits

- Max 5 concurrent jobs
- Regular users: N ≤ 5
- Admin users: N ≤ 6
- Memory estimated per job before accepting (N=6 worst case ~900MB)
- Worker instance: 4GB RAM, 2 shared vCPUs

## Environment

- `AT_HOME`: absent = local mode (no S3 backup), `DO` = deployed (S3 enabled)
- `API_HOST`: set by DO for worker → API internal URL
- `S3_REGION`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`: Spaces credentials (encrypted on DO)
