# Deployment & CI/CD

## Local Development

Run all components with one command:

```bash
robot2048
```

This starts:
- API on `http://localhost:8000` (FastAPI + SQLite)
- Julia worker polling localhost:8000
- UI dev server on `http://localhost:5173`

Ctrl+C stops all processes. The script is at project root (`robot2048`), symlinked to `~/.local/bin/`.

## Production (DigitalOcean)

### App Platform

- **App ID**: `90a1e251-4d6f-4ad1-aea7-285e3fc86b10`
- **Region**: FRA1 (Frankfurt)
- **URL**: `king-prawn-app-3givf.ondigitalocean.app`
- **Domain**: `robot2048.com`

### Components

| Component | Type | Instance | Cost |
|-----------|------|----------|------|
| API | Web Service | basic-xxs (512MB) | $5/mo |
| Julia Worker | Worker | basic-m (4GB/2vCPU) | $40/mo |
| UI | Static Site | — | free |

### Routing (Ingress)

| Path | Component |
|------|-----------|
| `/` | UI (static site) |
| `/service` | API (FastAPI) |

### Environment Variables (App-Level)

| Key | Value | Encrypted |
|-----|-------|-----------|
| `AT_HOME` | `DO` | no |
| `S3_REGION` | `fra1` | no |
| `S3_ACCESS_KEY` | (Spaces key) | yes |
| `S3_SECRET_KEY` | (Spaces secret) | yes |

Worker also gets `API_HOST` set to the API's private URL for internal communication.

### S3 Storage (Spaces)

- **Space**: `robot-2048` in `fra1`
- **Contents**: SQLite backups (`robot2048.db`), agent weights (`{name}.pkl`)
- **Credentials**: `~/aws/do_credentials.json` locally

## CI/CD

Push to `master` on `abachurin/robot-2048` → DigitalOcean auto-deploys all components (`deploy_on_push: true`).

No GitHub Actions — DO App Platform handles builds directly via GitHub webhook.

### Build Details

- **API**: Dockerfile → `python:3.11-slim-buster`, pip install, runs `main.py`
- **Julia Worker**: Dockerfile → `julia:1.11-bookworm`, Pkg.instantiate, runs `main.jl`
- **UI**: Node.js buildpack → `npm run build`, serves `dist/`

## CLI Tools

```bash
# App management
doctl apps list
doctl apps spec get <app-id>
doctl apps update <app-id> --spec <file>
doctl apps logs <app-id> --component <name> --follow

# GitHub
gh repo view abachurin/robot-2048

# DNS
doctl compute domain records list robot2048.com
```
