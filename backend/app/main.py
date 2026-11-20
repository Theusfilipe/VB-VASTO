import asyncio
import importlib.util
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.manager import SNAPSHOT_RATE, SimulationManager
from backend.app.schemas import (
    OperationResult,
    PolicyCatalog,
    PolicyNameRequest,
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
    try:
        return await manager_from(request).start()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/simulation/pause", response_model=SimulationSnapshot)
async def pause_simulation(request: Request) -> SimulationSnapshot:
    try:
        return await manager_from(request).pause()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/simulation/reset", response_model=SimulationSnapshot)
async def reset_simulation(body: ResetRequest, request: Request) -> SimulationSnapshot:
    try:
        return await manager_from(request).reset(body.seed)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/simulation/step", response_model=OperationResult)
async def step_simulation(body: StepRequest, request: Request) -> OperationResult:
    try:
        return await manager_from(request).step(body.action)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/simulation/speed", response_model=SimulationSnapshot)
async def set_speed(body: SpeedRequest, request: Request) -> SimulationSnapshot:
    try:
        return await manager_from(request).set_speed(body.speed)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def local_policy_path(root: Path, model_name: str) -> Path | None:
    model_path = (root / model_name).resolve()
    if not model_path.is_relative_to(root) or not model_path.is_file():
        return None
    config = model_path.with_name("config.json").resolve()
    if not config.is_relative_to(root) or not config.is_file():
        return None
    return model_path


def discover_policies(root: Path) -> tuple[str, ...]:
    names = []
    for path in root.rglob("*.zip"):
        name = path.relative_to(root).as_posix()
        try:
            PolicyRequest(model_name=name)
        except ValueError:
            continue
        if local_policy_path(root, name) is not None:
            names.append(name)
    return tuple(sorted(names))


def read_policy_names(root: Path) -> dict[str, str]:
    path = root / ".policy-names.json"
    if not path.exists():
        return {}
    try:
        names = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail=f"Could not read policy names: {error}") from error
    if not isinstance(names, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or not value.strip()
        for key, value in names.items()
    ):
        raise HTTPException(status_code=503, detail="Policy names file must contain a JSON object of non-empty names")
    return names


def save_policy_name(root: Path, model_name: str, display_name: str) -> None:
    names = read_policy_names(root)
    names[model_name] = display_name
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, prefix=".policy-names-", suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            json.dump(names, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
        os.replace(temporary_path, root / ".policy-names.json")
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"Could not save policy name: {error}") from error
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


@app.get("/simulation/policies", response_model=PolicyCatalog)
async def list_policies() -> PolicyCatalog:
    models_directory = os.getenv("ADAPTIVE_AI_MODELS_DIR")
    if not models_directory:
        return PolicyCatalog(configured=False)
    root = Path(models_directory).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise HTTPException(status_code=503, detail="ADAPTIVE_AI_MODELS_DIR must be a directory")
    policies = await asyncio.to_thread(discover_policies, root)
    names = await asyncio.to_thread(read_policy_names, root)
    return PolicyCatalog(
        configured=True,
        policies=policies,
        names={policy: names[policy] for policy in policies if policy in names},
    )


@app.post("/simulation/policies/name", response_model=PolicyCatalog)
async def set_policy_name(body: PolicyNameRequest) -> PolicyCatalog:
    models_directory = os.getenv("ADAPTIVE_AI_MODELS_DIR")
    if not models_directory:
        raise HTTPException(status_code=503, detail="Set ADAPTIVE_AI_MODELS_DIR to manage trusted local policies")
    root = Path(models_directory).expanduser().resolve()
    if local_policy_path(root, body.model_name) is None:
        raise HTTPException(status_code=404, detail="Policy and its config.json must be in ADAPTIVE_AI_MODELS_DIR")
    await asyncio.to_thread(save_policy_name, root, body.model_name, body.display_name)
    policies = await asyncio.to_thread(discover_policies, root)
    names = await asyncio.to_thread(read_policy_names, root)
    return PolicyCatalog(
        configured=True,
        policies=policies,
        names={policy: names[policy] for policy in policies if policy in names},
    )


@app.post("/simulation/policy", response_model=SimulationSnapshot)
async def load_policy(body: PolicyRequest, request: Request) -> SimulationSnapshot:
    models_directory = os.getenv("ADAPTIVE_AI_MODELS_DIR")
    if not models_directory:
        raise HTTPException(status_code=503, detail="Set ADAPTIVE_AI_MODELS_DIR to enable trusted local policies")
    root = Path(models_directory).expanduser().resolve()
    model_path = local_policy_path(root, body.model_name)
    if model_path is None:
        raise HTTPException(status_code=404, detail="Policy and its config.json must be in ADAPTIVE_AI_MODELS_DIR")
    try:
        return await manager_from(request).load_policy(model_path, body.model_name)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ImportError as error:
        raise HTTPException(status_code=503, detail="Install the learning extra to load PPO policies") from error
    except (OSError, ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def remove_policy_checkpoint(root: Path, model_name: str) -> None:
    checkpoint = root / model_name
    try:
        resolved = local_policy_path(root, model_name)
        if resolved is None:
            raise HTTPException(status_code=404, detail="Saved policy checkpoint not found")
        if resolved != checkpoint or checkpoint.is_symlink():
            raise HTTPException(status_code=409, detail="Cannot delete a policy through a linked path")
        checkpoint.unlink()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Saved policy checkpoint not found") from error
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"Could not delete the policy checkpoint: {error}") from error


@app.post("/simulation/policies/delete", response_model=PolicyRequest)
async def delete_policy(body: PolicyRequest) -> PolicyRequest:
    models_directory = os.getenv("ADAPTIVE_AI_MODELS_DIR")
    if not models_directory:
        raise HTTPException(status_code=503, detail="Set ADAPTIVE_AI_MODELS_DIR to manage trusted local policies")
    root = Path(models_directory).expanduser().resolve()
    await asyncio.to_thread(remove_policy_checkpoint, root, body.model_name)
    return body


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
        return await manager_from(request).start_live_training(
            body.total_steps, body.seed, root,
            total_episodes=body.total_episodes, max_episode_steps=body.max_episode_steps,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"Could not start training: {error}") from error


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
