import time

from fastapi.testclient import TestClient
import pytest
from pytest import approx

from backend.app.main import app
from backend.app.schemas import PolicyRequest


def test_snapshot_controls_and_shared_websocket_state() -> None:
    with TestClient(app) as client:
        initial = client.get("/simulation/state")
        assert initial.status_code == 200
        snapshot = initial.json()
        assert snapshot["status"] == "paused"
        assert snapshot["dimensions"] == {"width": 5, "height": 5}
        assert len(snapshot["agents"]) == 1
        assert snapshot["resources"]
        assert client.get("/simulation/config").json()["metabolism"]["max_energy"] == 36

        assert client.post("/simulation/speed", json={"speed": 5}).json()["speed"] == 5
        assert client.post("/simulation/speed", json={"speed": 3}).status_code == 422
        assert client.post("/simulation/start").json()["status"] == "running"
        assert client.post("/simulation/step", json={"action": 0}).status_code == 409
        assert client.post("/simulation/pause").json()["status"] == "paused"
        client.post("/simulation/reset", json={"seed": 12})

        step = client.post("/simulation/step", json={"action": 7})
        assert step.status_code == 200
        assert step.json()["snapshot"]["simulation_step"] == 1
        assert step.json()["reward"] is not None
        assert client.post("/simulation/step", json={"action": 8}).status_code == 422

        with client.websocket_connect("/ws/simulation") as websocket:
            observed = websocket.receive_json()
            assert observed["simulation_step"] == 1
            assert observed["agents"][0]["current_action"] == "SLEEP"
            websocket.receive_json()
            current = client.get("/simulation/state").json()
            assert current["simulation_step"] == 1

        reset = client.post("/simulation/reset", json={"seed": 42})
        assert reset.status_code == 200
        assert reset.json()["simulation_step"] == 0
        assert reset.json()["agents"][0]["position"] == client.post(
            "/simulation/reset", json={"seed": 42},
        ).json()["agents"][0]["position"]


def test_running_simulation_advances_independently_of_websocket_clients() -> None:
    with TestClient(app) as client:
        client.post("/simulation/reset", json={"seed": 9})
        client.post("/simulation/start")
        first = client.get("/simulation/state").json()["simulation_step"]
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.receive_json()
            websocket.receive_json()
            later = client.get("/simulation/state").json()["simulation_step"]
            assert later > first
        paused = client.post("/simulation/pause").json()
        step = paused["simulation_step"]
        assert client.get("/simulation/state").json()["simulation_step"] == step


def test_episode_metrics_are_emitted_from_completed_simulation() -> None:
    with TestClient(app) as client:
        client.post("/simulation/reset", json={"seed": 3})
        for _ in range(36):
            transition = client.post("/simulation/step", json={"action": 0}).json()
        assert transition["terminated"] is True
        metric = transition["snapshot"]["metrics"][-1]
        assert metric["episode"] == transition["snapshot"]["episode"]
        assert metric["survival_duration"] == 36
        assert metric["food_consumption"] == 0
        assert metric["sleeping_frequency"] == 0
        assert metric["reward"] == approx(-0.5)


def test_snapshot_is_immutable() -> None:
    with TestClient(app) as client:
        snapshot = client.app.state.simulation
        snapshot = snapshot._snapshot()
        try:
            snapshot.status = "running"
        except Exception as error:
            assert error.__class__.__name__ == "ValidationError"
        else:
            raise AssertionError("snapshot models must be immutable")
        try:
            snapshot.agents[0].energy = 0
        except Exception as error:
            assert error.__class__.__name__ == "ValidationError"
        else:
            raise AssertionError("agent snapshots must be immutable")


def test_training_requires_optional_dependencies_and_configured_output(monkeypatch) -> None:
    monkeypatch.delenv("ADAPTIVE_AI_MODELS_DIR", raising=False)
    with TestClient(app) as client:
        assert client.post("/training/start", json={"total_steps": 511}).status_code == 422
        assert client.post("/training/start", json={"total_steps": 512}).status_code == 503


def test_live_training_launch_failure_keeps_previous_simulation_state(tmp_path, monkeypatch) -> None:
    import backend.app.manager as manager_module

    async def fail_to_start(*args, **kwargs):
        raise OSError("process launch failed")

    monkeypatch.setattr(manager_module.asyncio, "create_subprocess_exec", fail_to_start)
    monkeypatch.setenv("ADAPTIVE_AI_MODELS_DIR", str(tmp_path))
    with TestClient(app) as client:
        response = client.post("/training/start", json={"total_steps": 512})
        assert response.status_code == 500
        snapshot = client.get("/simulation/state").json()
        assert snapshot["execution_mode"] == "simulation"
        assert snapshot["status"] == "paused"
        assert client.post("/training/stop").json()["execution_mode"] == "simulation"


def test_policy_paths_allow_training_runs_but_reject_traversal() -> None:
    assert PolicyRequest(model_name="live-a1b2c3/ppo_final.zip").model_name
    for model_name in ("../outside.zip", "live/../../outside.zip", "/outside.zip", "bad name.zip"):
        try:
            PolicyRequest(model_name=model_name)
        except ValueError:
            continue
        raise AssertionError(f"unsafe policy path was accepted: {model_name}")


def test_policy_catalog_lists_only_local_checkpoints_with_config(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ADAPTIVE_AI_MODELS_DIR", raising=False)
    with TestClient(app) as client:
        assert client.get("/simulation/policies").json() == {"configured": False, "policies": []}
        root = tmp_path / "models"
        monkeypatch.setenv("ADAPTIVE_AI_MODELS_DIR", str(root))
        assert client.get("/simulation/policies").json() == {"configured": True, "policies": []}

        run = root / "live-example"
        run.mkdir(parents=True)
        (root / "unpaired.zip").touch()
        (run / "config.json").write_text("{}", encoding="utf-8")
        for name in ("ppo_final.zip", "ppo_512_steps.zip", "bad name.zip"):
            (run / name).touch()
        assert client.get("/simulation/policies").json() == {
            "configured": True,
            "policies": ["live-example/ppo_512_steps.zip", "live-example/ppo_final.zip"],
        }
        assert client.post("/simulation/policy", json={"model_name": "unpaired.zip"}).status_code == 404


def test_live_training_streams_snapshots_and_loads_saved_policy(tmp_path, monkeypatch) -> None:
    pytest.importorskip("stable_baselines3")
    monkeypatch.setenv("ADAPTIVE_AI_MODELS_DIR", str(tmp_path))
    with TestClient(app) as client:
        started = client.post("/training/start", json={"total_steps": 512, "seed": 17})
        assert started.status_code == 200, started.text
        initial = started.json()
        assert initial["execution_mode"] == "training"
        assert initial["status"] == "running"
        run_name = initial["training_run"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            latest = client.get("/simulation/state").json()
            if latest["status"] == "paused":
                break
            time.sleep(0.05)
        else:
            client.post("/training/stop")
            pytest.fail("live PPO run did not finish")

        assert latest["execution_mode"] == "training"
        assert latest["training_timesteps"] >= 512
        assert (tmp_path / run_name / "ppo_final.zip").is_file()
        assert (tmp_path / run_name / "config.json").is_file()

        checkpoint = f"{run_name}/ppo_final.zip"
        assert checkpoint in client.get("/simulation/policies").json()["policies"]
        loaded = client.post(
            "/simulation/policy",
            json={"model_name": checkpoint},
        )
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["execution_mode"] == "evaluation"
        assert loaded.json()["policy_name"] == checkpoint

        restarted = client.post("/training/start", json={"total_steps": 5_000_000, "seed": 19})
        assert restarted.status_code == 200, restarted.text
        assert client.post("/simulation/start").status_code == 409
        assert client.post("/simulation/pause").status_code == 409
        assert client.post("/simulation/reset", json={"seed": 20}).status_code == 409
        assert client.post("/simulation/speed", json={"speed": 2}).status_code == 409
        assert client.post("/simulation/policy", json={
            "model_name": f"{run_name}/ppo_final.zip",
        }).status_code == 409
        stopped = client.post("/training/stop")
        assert stopped.status_code == 200, stopped.text
        assert stopped.json()["execution_mode"] == "training"
        assert stopped.json()["status"] == "paused"
        assert (tmp_path / stopped.json()["training_run"] / "ppo_final.zip").is_file()
        assert client.post("/simulation/pause").json()["policy_name"] == checkpoint
        assert client.get("/simulation/state").json()["execution_mode"] == "evaluation"
