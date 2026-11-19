# VB-VASTO
VB-VASTO (Virtual Biology — Vision, Audition, Somatic, Taste, Olfaction) is an artificial life framework exploring the emergence of autonomous animal behavior through sensory perception, deep learning, reinforcement learning, and neuroevolution in simulated ecosystems.

The initial `adaptive_ai` release is a single-rabbit Gymnasium simulation and a
Stable-Baselines3 PPO workflow. Neither the rabbit nor the simulation depends on
a game engine, a policy, or a display.

## Install

Use Python 3.12 or newer. From the repository root, activate an existing virtual
environment or create one with `python -m venv .venv` and `source .venv/bin/activate`.
Then install the package with:

```sh
python -m pip install -e '.[learning,visualization,test]'
python -m pytest
```

For simulation only, install with `python -m pip install -e .`. For headless PPO
you can omit the `visualization` extra and the `--plot` and `--render` flags.

## Train, resume, and evaluate

```sh
python train.py --steps 150000 --seed 0 --output runs/rabbit --plot
python evaluate.py runs/rabbit/ppo_final.zip --episodes 20 --seed 0
python evaluate.py runs/rabbit/ppo_final.zip --episodes 1 --render
python train.py --resume runs/rabbit/ppo_final.zip --steps 50000 --seed 1 --output runs/rabbit-resumed --plot
```

Each invocation of `train.py` requires a new output directory. It checks the
default environment with Gymnasium and the configured environment with SB3,
then trains a CPU `MlpPolicy`. The
requested step count is a minimum because PPO finishes its current rollout.
The output includes `progress.csv` (PPO statistics), `monitor.csv` (episode
reward, duration, food eaten, successful moves, and sleep steps), periodic
checkpoints (`--checkpoint-every`, default 25,000; saved after completed PPO
updates), and `ppo_final.zip`.
`--plot` saves `training.png` with reward, survival duration, food eaten, and
sleep frequency (sleep steps divided by episode length).

Keep the generated `config.json` beside the checkpoint when moving it.
Resume restores the model, optimizer and timestep count, but begins a fresh
environment trajectory and logs to a new directory. Only load checkpoints you
trust: SB3's model format can deserialize Python objects.
Evaluation uses deterministic predictions without updating weights, and
independent environment seeds starting at `2**32 + --seed`. It reports per-episode
and mean survival length, food eaten, successful moves, sleep steps, and total
reward. `--render` needs a graphical Matplotlib backend; library users can use
`render_mode="rgb_array"` for headless frames.

In one default training run (seed 0, 150,000 requested steps), the policy
reached the 200-step limit in 16 of 20 held-out episodes, averaging 192.7
survival steps, 19.35 food units eaten, 111.45 successful moves, and 40.7 sleep
steps per episode. Results can vary by seed and configuration.

## Simulation and configuration

By default, the world is a 5 × 5 grid with 8 plant cells and a 200-step time
limit. Reset samples the starting cells and rabbit position using Gymnasium's
seeded random generator. Below the target plant count (`ceil(width × height ×
resource_density)`), each tick has a 25% chance to regenerate one plant at a
uniformly sampled empty cell. The rabbit's opaque UUID is deliberately
independent of the simulation seed and never enters policy observations.

Supply `train.py --config path/to/config.json` to override defaults. Fields map
to `RabbitSurvivalConfig`, with nested `metabolism` and `rewards` objects.
Omitted fields use their defaults. For example:

```json
{
  "width": 6,
  "height": 5,
  "resource_density": 0.4,
  "regeneration_probability": 0.2,
  "perception_radius": 2,
  "metabolism": {"max_energy": 40},
  "rewards": {"invalid": -0.2}
}
```

The actions in order are `IDLE`, `MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`,
`MOVE_RIGHT`, `COLLECT`, `EAT`, and `SLEEP`. Each advances one tick. Movement and
collection cost extra energy and fatigue, and are blocked at or above 85% of
maximum fatigue by default. Collection stores one Plant Fiber in inventory;
eating uses one stored unit, adding up to its nutritional value after awake
metabolism without directly reducing fatigue. Sleep recovers fatigue but
still consumes energy. An impossible action is an awake no-op with a penalty.
Starvation terminates the episode; reaching the step limit while alive
truncates it.

The observation is a fixed `float32` vector bounded by 0 and 1:

1. Energy/max energy, fatigue/max fatigue, inventory count/capacity (zero for
   zero-capacity inventory), and current-cell plant presence.
2. Left, right, top, and bottom boundary distances, divided by the respective
   axis length minus one (zero on singleton axes).
3. Binary plant presence at relative offsets within Manhattan distance
   `perception_radius`, excluding the center, ordered by increasing `dy` then
   increasing `dx`. Off-grid offsets are zero. At radius 1 this is up, left,
   right, down. No distant resource locations are observed.

The default reward is **+0.10** on a tick survived or **−4.00** on starvation,
plus **+0.025** per unit of net energy gained from eating and **−0.15** for an
invalid action. Movement, collecting, and sleeping grant no action bonus.
Energy caps prevent a nutrition bonus for eating at full energy. Inventory is
limited, and even a rabbit that only sleeps runs out of energy after 72 ticks
under the default settings.

## Architecture

- `core/` holds entity and transition types. `agents/` composes rabbit state
  from reusable `mechanics/` (metabolism and inventory) and `resources/`.
- `environments/rabbit_survival/` owns the grid and the rules for executing
  actions, producing observations, and scoring transitions. Its Gymnasium
  adapter connects those modules without selecting actions.
- `learning/` trains numerical Gymnasium environments using SB3. PPO manages
  rollout collection and optimization; the scripts select the environment
  and handle run configuration. `visualization/` consumes simulation state and
  monitor data without changing them.

New environments can provide their own sensors, actions, and rewards without
changing PPO. External game adapters should implement the same environment
interface. Multiple agents, external transports, and other learning algorithms
are outside this milestone.

## Web simulation dashboard

The optional dashboard is split into `/backend` and `/frontend`. It observes the
same headless Gymnasium environment used by the Python tools; the API does not
reimplement simulation mechanics, and the Python package remains usable without
FastAPI or Node.js.

### Install and run locally

Use the Python 3.12 environment from the setup above, then install the web extra:

```sh
python -m pip install -e '.[web]'
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
cd frontend
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The API allows requests
from `http://localhost:3000` and `http://127.0.0.1:3000`; set
`DASHBOARD_ORIGINS` to a comma-separated list to change those origins. Set
`NEXT_PUBLIC_API_URL` before starting Next.js to change the backend HTTP URL
(the WebSocket URL is derived from it).

Install `.[learning,web]` instead of `.[web]` to enable PPO policy evaluation
and live training. Set `ADAPTIVE_AI_MODELS_DIR` to a directory for trusted
checkpoints and training runs before starting the backend. To evaluate an
existing policy, enter a checkpoint filename such as `ppo_final.zip` and select
**Load policy**. Checkpoints in a live training subdirectory can be loaded by
their relative path, such as `live-a1b2c3d4e5f6/ppo_final.zip`. PPO checkpoints
use Python serialization; load only files you trust. Without a loaded policy,
**Start** advances the environment with `IDLE`; manual stepping allows
selecting an action.

Select **Train live** while the simulation is paused to start a separate PPO
training environment and stream its snapshots to the dashboard. The dashboard
shows the active training environment and completed-episode metrics; snapshots
are emitted at most every 100 ms rather than on every training transition.
Training runs and checkpoints are saved under `ADAPTIVE_AI_MODELS_DIR`; select
**Stop training** to request a graceful stop and save the final checkpoint.
Ordinary `train.py` remains headless and independent of the web service.

### API and execution model

- `GET /simulation/state` returns the latest immutable snapshot;
  `GET /simulation/config` returns the current environment configuration.
- `POST /simulation/start`, `/simulation/pause`, `/simulation/reset`,
  `/simulation/step`, and `/simulation/speed` control the shared simulation.
  Reset accepts an optional JSON `seed`; step takes an action integer from 0 to
  7 and is rejected with HTTP 409 while running; speed accepts `1`, `2`, `5`,
  or `10`.
- `POST /simulation/policy` loads a checkpoint by filename from the configured
  local models directory and adopts its saved adjacent `config.json`. Candidate
  policies load off the event loop, and are installed only after their Gym
  observation and action spaces match the simulation.
- `POST /training/start` accepts `total_steps` and `seed` and starts an isolated
  live PPO run; `POST /training/stop` requests a cooperative stop. Live training
  requires both the `learning` extra and `ADAPTIVE_AI_MODELS_DIR`.
- `/ws/simulation` sends current snapshots at 10 Hz. It does not own or advance
  the environment; every connected browser observes the same manager. The
  manager runs independently of connected clients, advances at
  `10 × selected speed` steps per second, and only retains the latest 100
  episode
  metrics. There is no queue of stale frames for slow clients.

Snapshots have dimensions, episode and step numbers, run status, resources,
agent state, cumulative reward, and completed episode metrics. The schema
represents agent and resource collections, although this simulation currently
creates one rabbit and Plant Fiber. Episode charts show only completed episode
results emitted by the simulation: reward, survival duration, rolling mean of
up to ten episode durations, food consumed, and the fraction of steps spent
sleeping. No synthetic training history is generated. The Canvas draws the
world and agent selection; the browser does not choose policy actions or set
simulation timing.

The dashboard frontend has its own `npm run lint`, `npx tsc --noEmit`, and
`npm run build` checks. Backend API, controls, snapshots, and WebSocket behavior
are exercised by `python -m pytest`.
