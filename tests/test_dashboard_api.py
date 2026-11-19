from fastapi.testclient import TestClient

from backend.app.main import app


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
            assert later >= first
        paused = client.post("/simulation/pause").json()
        step = paused["simulation_step"]
        assert client.get("/simulation/state").json()["simulation_step"] == step
