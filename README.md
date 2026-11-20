# VB-VASTO

VB-VASTO (Virtual Biology — Vision, Audition, Somatic, Taste, Olfaction) is an
artificial life project built around a single-rabbit Gymnasium simulation. It
includes a Stable-Baselines3 PPO workflow and an optional web dashboard. The
simulation itself does not need a game engine, a policy, or a display.

## Install

Use Python 3.12 or newer. From the repository root, activate an existing
virtual environment or create one as shown below.

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `python` already works on Windows, you can also use `python -m venv .venv`.
If it opens the Microsoft Store, use `py` to create the environment.

**macOS or Linux:**

```sh
python3 -m venv .venv
source .venv/bin/activate
```

With the environment active, **pick one** of these blocks to install the
package and run the tests.

**Using `python`:**

```sh
python -m pip install -e '.[learning,visualization,test]'
python -m pytest
```

**Using `py` (Windows):**

```powershell
py -m pip install -e '.[learning,visualization,test]'
py -m pytest
```

For simulation only, use `python -m pip install -e .` or `py -m pip install -e .`
on Windows. For headless PPO you can omit the `visualization` extra and the
`--plot` and `--render` flags.

## Train, resume, and evaluate

Choose the block matching your Python command.

**Using `python`:**

```Shell
python train.py --steps 150000 --seed 0 --output runs/rabbit --plot
python evaluate.py runs/rabbit/ppo_final.zip --episodes 20 --seed 0
python evaluate.py runs/rabbit/ppo_final.zip --episodes 1 --render
python train.py --resume runs/rabbit/ppo_final.zip --steps 50000 --seed 1 --output runs/rabbit-resumed --plot
```

**Using `py` (Windows):**

```powershell
py train.py --steps 150000 --seed 0 --output runs\rabbit --plot
py evaluate.py runs\rabbit\ppo_final.zip --episodes 20 --seed 0
py evaluate.py runs\rabbit\ppo_final.zip --episodes 1 --render
py train.py --resume runs\rabbit\ppo_final.zip --steps 50000 --seed 1 --output runs\rabbit-resumed --plot
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
seeded random generator. Below the target plant count (`ceil(width × height × resource_density)`), each tick has a 25% chance to regenerate one plant at a
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

- `core/` holds shared entity and observation types. `agents/` composes rabbit state
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

On Windows, after installing the Python `web` extra and running `npm ci` in
`frontend` once, start both dev servers from the repository root:

```powershell
.\start-dashboard.ps1
```

The script opens a separate window for each server. Open
[http://127.0.0.1:3000](http://127.0.0.1:3000) once both servers are ready.
Press Ctrl+C in each server window to stop it. The launcher does not install
anything and defaults `ADAPTIVE_AI_MODELS_DIR` to the repository's `runs`
directory unless you set it yourself.

#### Manual startup

From the repository root, use the Python environment from the setup above.
If the web dependencies are not already installed in that environment, install
the `web` extra before starting the backend. Pick one of these blocks:

**Using `py` (Windows PowerShell):**

```powershell
py -m pip install -e '.[web]'
py -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

**Using `python`:**

```sh
python -m pip install -e '.[web]'
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Running Uvicorn as a Python module works even if the `uvicorn` command is not on
your `PATH`.

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

Install `.[learning,web]` instead of `.[web]` to enable running PPO policies
and live training. Before starting the backend, set `ADAPTIVE_AI_MODELS_DIR`
to a trusted directory for checkpoints and training runs. For example, in
PowerShell from the repository root: `$env:ADAPTIVE_AI_MODELS_DIR = ".\runs"`.

- **Manual control:** Pause the simulation, choose an action, and click **Step once**.
  The selected action is used instead of the policy. **Reset world** starts a
  new episode.
- **Run simulation:** Choose a checkpoint from **Saved policy**, click **Load policy**,
  then **Run policy**. This does not train the policy. If no policy is loaded,
  **Run idle simulation** advances the world using `IDLE` on every step.
  Use the **Policies** tab to name or rename saved checkpoints; these labels are
  stored in `.policy-names.json` in the models directory and do not rename or
  modify checkpoint files.
  To permanently remove a saved checkpoint, click **Delete** and confirm in the
  popup. Only the selected `.zip` is removed; other checkpoints and their
  shared `config.json` stay in place. A policy already loaded in memory can
  still be used until it is replaced or the backend restarts.
  To add an existing checkpoint, place the `.zip` and its `config.json` together
  in the models directory or a subdirectory, then click **Refresh policies**.
  PPO checkpoints use Python serialization; only add files you trust.
- **Training:** While the simulation is paused, choose a budget in **Steps** or
  **Episodes** and click **Start live training**. **Max steps per episode**
  sets the time limit for this run only; leave it blank to use the current
  world's limit. An episode can still end early if the rabbit starves.
  The dashboard shows both training steps and completed episodes. A step
  budget is a minimum because PPO finishes its current rollout. An episode
  budget stops at the requested count; PPO does not learn from the unfinished
  rollout, so very short runs may end before the first update.
  Training uses a separate environment and streams snapshots at most every 100 ms.
  **Stop training** requests a graceful stop. When the run saves its final
  checkpoint, the policy list refreshes and selects it for loading. You can
  also choose **Return to simulation** to go back without loading it.

Ordinary `train.py` remains headless and independent of the web service.

### API and execution model

- `GET /simulation/state` returns the latest immutable snapshot;
  `GET /simulation/config` returns the current environment configuration.
- `PATCH /simulation/config/metabolism` changes just the supplied metabolic
  parameters between simulation steps; for example, `{"basal_energy_cost": 2}`
  consumes two energy units per awake step before action costs. Omitted values
  keep their current settings. `POST /simulation/config/metabolism/reset` restores
  the metabolic defaults without resetting the episode. Lowering a maximum
  clamps the current value; raising it does not refill energy or induce fatigue.
  Initial fractions apply at the next world reset. Both operations preserve the
  loaded policy and history. They are unavailable in the training view: stop
  training and return to the simulation first. Settings persist across episode
  resets and are saved with subsequent training runs, not to existing checkpoints.
- `POST /simulation/start`, `/simulation/pause`, `/simulation/reset`,
  `/simulation/step`, and `/simulation/speed` control the shared simulation.
  Reset accepts an optional JSON `seed`; step takes an action integer from 0 to
  7 and is rejected with HTTP 409 while running; speed accepts `1`, `2`, `5`,
  or `10`.
- `GET /simulation/policies` lists checkpoint names relative to the configured
  models directory without loading them, including any saved display names.
- `POST /simulation/policies/name` saves a display name for a checkpoint without
  changing its filename or contents.
- `POST /simulation/policies/delete` accepts a relative `model_name` and
  permanently removes only that saved checkpoint.
- `POST /simulation/policy` loads a checkpoint by relative filename from the configured
  local models directory and adopts its saved adjacent `config.json`. Candidate
  policies load off the event loop, and are installed only after their Gym
  observation and action spaces match the simulation.
- `POST /training/start` accepts either `total_steps` or `total_episodes`,
  plus optional `max_episode_steps` and `seed`, and starts an isolated
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
