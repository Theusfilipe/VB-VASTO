import asyncio
import importlib.util
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.manager import SNAPSHOT_RATE, SimulationManager
from backend.app.schemas import (
    OperationResult,
    PolicyRequest,
    ResetRequest,
    SimulationConfig,
    SimulationSnapshot,
    SpeedRequest,
    StepRequest,
    TrainingRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    manager = SimulationManager()
    app.state.simulation = manager
    await manager.start_runner()
    try:
        yield
    finally:
        await manager.close()


app = FastAPI(title="Adaptive AI Simulation API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("DASHBOARD_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def manager_from(request: Request | WebSocket) -> SimulationManager:
    return request.app.state.simulation


@app.get("/simulation/state", response_model=SimulationSnapshot)
async def get_state(request: Request) -> SimulationSnapshot:
    return await manager_from(request).snapshot()


@app.get("/simulation/config", response_model=SimulationConfig)
async def get_config(request: Request) -> SimulationConfig:
    return SimulationConfig.model_validate(await manager_from(request).configuration())


@app.post("/simulation/start", response_model=SimulationSnapshot)
async def start_simulation(request: Request) -> SimulationSnapshot:
    return await manager_from(request).start()


@app.post("/simulation/pause", response_model=SimulationSnapshot)
async def pause_simulation(request: Request) -> SimulationSnapshot:
    return await manager_from(request).pause()


@app.post("/simulation/reset", response_model=SimulationSnapshot)
async def reset_simulation(body: ResetRequest, request: Request) -> SimulationSnapshot:
    return await manager_from(request).reset(body.seed)


@app.post("/simulation/step", response_model=OperationResult)
async def step_simulation(body: StepRequest, request: Request) -> OperationResult:
    try:
        return await manager_from(request).step(body.action)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/simulation/speed", response_model=SimulationSnapshot)
async def set_speed(body: SpeedRequest, request: Request) -> SimulationSnapshot:
    return await manager_from(request).set_speed(body.speed)


@app.post("/simulation/policy", response_model=SimulationSnapshot)
async def load_policy(body: PolicyRequest, request: Request) -> SimulationSnapshot:
    models_directory = os.getenv("ADAPTIVE_AI_MODELS_DIR")
    if not models_directory:
        raise HTTPException(status_code=503, detail="Set ADAPTIVE_AI_MODELS_DIR to enable trusted local policies")
    root = Path(models_directory).expanduser().resolve()
    model_path = (root / body.model_name).resolve()
    if not model_path.is_relative_to(root) or not model_path.is_file():
        raise HTTPException(status_code=404, detail="Policy not found in ADAPTIVE_AI_MODELS_DIR")
    try:
        return await manager_from(request).load_policy(model_path)
    except ImportError as error:
        raise HTTPException(status_code=503, detail="Install the learning extra to load PPO policies") from error
    except (OSError, ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/training/start", response_model=SimulationSnapshot)
async def start_live_training(body: TrainingRequest, request: Request) -> SimulationSnapshot:
    if importlib.util.find_spec("stable_baselines3") is None:
        raise HTTPException(status_code=503, detail="Install the learning extra to start PPO training")
    models_directory = os.getenv("ADAPTIVE_AI_MODELS_DIR")
    if not models_directory:
        raise HTTPException(status_code=503, detail="Set ADAPTIVE_AI_MODELS_DIR to save trusted training runs")
    root = Path(models_directory).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
        return await manager_from(request).start_live_training(body.total_steps, body.seed, root)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=500, detail="Unable to create the training output directory") from error


@app.post("/training/stop", response_model=SimulationSnapshot)
async def stop_live_training(request: Request) -> SimulationSnapshot:
    return await manager_from(request).stop_live_training()


@app.websocket("/ws/simulation")
async def simulation_events(websocket: WebSocket) -> None:
    await websocket.accept()
    manager = manager_from(websocket)
    try:
        while True:
            snapshot = await manager.snapshot()
            await websocket.send_json(snapshot.model_dump(mode="json"))
            await asyncio.sleep(1 / SNAPSHOT_RATE)
    except WebSocketDisconnect:
        return
