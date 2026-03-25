"""
Worker API endpoints — called by the Julia worker over HTTP.
These are internal endpoints, not exposed to the frontend.
"""
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Response
from base.utils import *

router = APIRouter(prefix="/worker", tags=["worker"])


# --- Jobs ---

@router.get('/jobs')
async def worker_jobs():
    """List all jobs with description, status, type."""
    return DB.get_all_jobs()


@router.post('/jobs/{description}/launch')
async def worker_launch_job(description: str):
    """Set job to RUN, return full job dict."""
    job = DB.launch_job(description)
    if job is None:
        return Response(status_code=404)
    return job


@router.get('/jobs/{description}/status')
async def worker_job_status(description: str):
    """Lightweight status check. Returns {"status": int} or 404."""
    status = DB.get_job_status(description)
    if status is None:
        return Response(status_code=404)
    return {"status": status}


@router.put('/jobs/{description}/timing')
async def worker_update_timing(description: str, elapsed: int, remaining: int):
    DB.update_job_timing(description, elapsed, remaining)


@router.put('/jobs/{description}/alpha')
async def worker_update_alpha(description: str, alpha: float):
    DB.update_job_alpha(description, alpha)


@router.delete('/jobs/{description}')
async def worker_delete_job(description: str):
    DB.delete_job(description)


# --- Agents ---

@router.get('/agents/{name}')
async def worker_get_agent(name: str):
    """Get full agent metadata."""
    agent = DB.get_agent(name)
    if agent is None:
        return Response(status_code=404)
    return agent


@router.put('/agents/{name}')
async def worker_update_agent(name: str, updates: dict):
    """Partial update of agent metadata."""
    DB.update_agent(name, updates)


@router.post('/agents/{name}/weights')
async def worker_upload_weights(name: str, file: UploadFile = File(...)):
    """Upload agent weights binary to S3."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        S3.upload(tmp_path, name)
    finally:
        os.remove(tmp_path)
    return {"status": "ok"}


@router.get('/agents/{name}/weights')
async def worker_download_weights(name: str):
    """Download agent weights binary from S3."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
        tmp_path = tmp.name
    found = S3.download(name, tmp_path)
    if not found:
        os.remove(tmp_path)
        return Response(status_code=404)
    with open(tmp_path, 'rb') as f:
        data = f.read()
    os.remove(tmp_path)
    return Response(content=data, media_type='application/octet-stream')


# --- Games ---

@router.post('/games')
async def worker_save_game(game: dict):
    """Save a game (test best game or watch game)."""
    DB.save_game(
        name=game['name'], user=game['user'],
        score=game['score'], num_moves=game['numMoves'],
        max_tile=game['maxTile'], initial=game['initial'],
        moves=game['moves'], tiles=game['tiles']
    )


@router.put('/games/{user}/moves')
async def worker_update_watch_game(user: str, data: dict):
    """Append moves and tiles to a watch game."""
    DB.update_watch_game(user, data['moves'], data['tiles'])


@router.delete('/games/{user}')
async def worker_delete_game(user: str):
    DB.delete_game(user)


# --- Logs ---

@router.post('/logs')
async def worker_add_log(data: dict):
    """Add a log entry for a user."""
    DB.add_log(data['userName'], data['text'])


# --- Watch ---

@router.put('/watch/{description}/loading')
async def worker_set_watch_loading(description: str, loading: bool):
    DB.set_watch_loading(description, loading)


# --- Cleanup ---

@router.post('/cleanup')
async def worker_cleanup():
    """Clean orphaned watch jobs/games on worker startup."""
    DB.clean_watch_jobs()
    return {"status": "ok"}
