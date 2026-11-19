"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { WorldCanvas } from "@/components/WorldCanvas";
import { TrainingMetrics } from "@/components/TrainingMetrics";
import { API_URL, WS_URL } from "@/lib/types";
import type { AgentSnapshot, SimulationConfig, SimulationSnapshot } from "@/lib/types";

const ACTIONS = ["IDLE", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "COLLECT", "EAT", "SLEEP"];
const speeds = [1, 2, 5, 10] as const;

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Backend returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function StatBar({ label, value, max, tone }: { label: string; value: number; max: number; tone: string }) {
  const percent = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div className="stat-block">
      <div className="stat-label"><span>{label}</span><b>{value.toFixed(1)} <small>/ {max}</small></b></div>
      <div className="progress-track"><div className={`progress-value ${tone}`} style={{ width: `${percent}%` }} /></div>
    </div>
  );
}

function Inspector({ agent, reward }: { agent: AgentSnapshot | undefined; reward: number }) {
  if (!agent) {
    return <div className="inspector-empty">Select an agent in the world to inspect its state.</div>;
  }
  return (
    <div className="inspector-content">
      <div className="agent-name"><span className="agent-avatar">R</span><div><b>Rabbit</b><span>{agent.alive ? "Alive" : "Deceased"}</span></div></div>
      <p className="agent-id" title={agent.identifier}>{agent.identifier}</p>
      <StatBar label="Energy" value={agent.energy} max={agent.max_energy} tone="energy" />
      <StatBar label="Fatigue" value={agent.fatigue} max={agent.max_fatigue} tone="fatigue" />
      <div className="inspector-grid">
        <div><span>Position</span><b>{agent.position.x}, {agent.position.y}</b></div>
        <div><span>Inventory</span><b>{agent.inventory} / {agent.inventory_capacity} fiber</b></div>
        <div><span>Current action</span><b>{agent.current_action.replaceAll("_", " ")}</b></div>
        <div><span>Age</span><b>{agent.age} steps</b></div>
        <div><span>Cumulative reward</span><b>{reward.toFixed(2)}</b></div>
        <div><span>Perception</span><b>{agent.perception_radius} cells</b></div>
      </div>
    </div>
  );
}

export function Dashboard() {
  const [snapshot, setSnapshot] = useState<SimulationSnapshot | null>(null);
  const [config, setConfig] = useState<SimulationConfig | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [action, setAction] = useState(0);
  const [seed, setSeed] = useState("0");
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const updateSnapshot = useCallback((next: SimulationSnapshot) => {
    setSnapshot(next);
    setSelectedAgent((current) =>
      current && next.agents.some((agent) => agent.identifier === current)
        ? current
        : next.agents[0]?.identifier ?? null,
    );
  }, []);

  useEffect(() => {
    let active = true;
    let retryTimer: ReturnType<typeof setTimeout>;
    let socket: WebSocket | null = null;
    let delay = 400;
    void Promise.all([
      request<SimulationSnapshot>("/simulation/state"),
      request<SimulationConfig>("/simulation/config"),
    ]).then(([state, settings]) => {
      if (!active) return;
      updateSnapshot(state);
      setConfig(settings);
      setError(null);
    }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : "Cannot reach the simulation API.");
    });

    const connect = () => {
      if (!active) return;
      socket = new WebSocket(WS_URL);
      socket.onopen = () => {
        if (!active) return;
        delay = 400;
        setConnected(true);
        setError(null);
      };
      socket.onmessage = (event) => {
        if (!active) return;
        try {
          updateSnapshot(JSON.parse(event.data) as SimulationSnapshot);
        } catch {
          setError("Received an invalid simulation snapshot.");
        }
      };
      socket.onerror = () => socket?.close();
      socket.onclose = () => {
        if (!active) return;
        setConnected(false);
        retryTimer = setTimeout(connect, delay);
        delay = Math.min(delay * 2, 5000);
      };
    };
    connect();
    return () => {
      active = false;
      clearTimeout(retryTimer);
      socket?.close();
    };
  }, [updateSnapshot]);

  const selected = useMemo(
    () => snapshot?.agents.find((agent) => agent.identifier === selectedAgent),
    [snapshot, selectedAgent],
  );

  async function control(path: string, body?: unknown) {
    setBusy(true);
    setError(null);
    try {
      const result = await request<SimulationSnapshot | { snapshot: SimulationSnapshot }>(path, body);
      updateSnapshot("snapshot" in result ? result.snapshot : result);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Simulation request failed.");
    } finally {
      setBusy(false);
    }
  }

  const running = snapshot?.status === "running";

  return (
    <main className="dashboard-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="Adaptive Field home"><span className="brand-mark">a.</span><span>adaptive<span className="brand-light">field</span></span></a>
        <div className="topbar-meta"><span className={`connection-dot ${connected ? "online" : ""}`} />{connected ? "Live connection" : "Reconnecting"}<i />Simulation dashboard</div>
      </header>

      <section className="page-intro">
        <div><p className="eyebrow">OBSERVATION STUDIO <span>•</span> RABBIT SURVIVAL</p><h1>Simulation dashboard</h1><p className="intro-copy">A live view into agent behavior and environmental state.</p></div>
        <div className="episode-badge"><span>EPISODE</span><b>{snapshot?.episode ?? "—"}</b><i />{running ? "RUNNING" : "PAUSED"}</div>
      </section>

      {error && <div className="error-banner" role="status"><span>Connection notice</span>{error}<button onClick={() => window.location.reload()}>Retry</button></div>}

      <section className="workspace">
        <div className="panel world-panel">
          <div className="panel-header"><div><p className="section-label">ENVIRONMENT</p><h2>World view</h2></div><div className="world-status"><span className="live-dot" />{running ? "LIVE" : "STATIC"}<i />{snapshot?.dimensions.width ?? "—"} × {snapshot?.dimensions.height ?? "—"}</div></div>
          {snapshot ? (
            <WorldCanvas
              width={snapshot.dimensions.width}
              height={snapshot.dimensions.height}
              agents={snapshot.agents}
              resources={snapshot.resources}
              selectedAgent={selectedAgent}
              onSelectAgent={setSelectedAgent}
            />
          ) : <div className="world-loading">Waiting for simulation snapshot…</div>}
          <div className="world-legend"><span><i className="legend-rabbit" />Rabbit</span><span><i className="legend-resource" />Plant fiber</span><span><i className="legend-radius" />Perception radius</span><span className="world-step">STEP <b>{snapshot?.simulation_step ?? "—"}</b></span></div>
          <div className="controls">
            <button className={`button ${running ? "button-muted" : "button-primary"}`} disabled={busy || running} onClick={() => void control("/simulation/start")}><span>▶</span> Start</button>
            <button className={`button ${running ? "button-primary" : "button-muted"}`} disabled={busy || !running} onClick={() => void control("/simulation/pause")}><span>Ⅱ</span> Pause</button>
            <button className="button button-outline" disabled={busy || running} onClick={() => void control("/simulation/step", { action })}><span>→</span> Step</button>
            <select className="action-select" aria-label="Action for manual step" value={action} onChange={(event) => setAction(Number(event.target.value))} disabled={busy || running}>{ACTIONS.map((name, index) => <option value={index} key={name}>{name.replaceAll("_", " ")}</option>)}</select>
            <div className="speed-control"><span>SPEED</span><div>{speeds.map((speed) => <button key={speed} onClick={() => void control("/simulation/speed", { speed })} className={snapshot?.speed === speed ? "speed-active" : ""} aria-pressed={snapshot?.speed === speed}>{speed}×</button>)}</div></div>
            <div className="reset-control"><input aria-label="Reset seed" value={seed} inputMode="numeric" onChange={(event) => setSeed(event.target.value)} /><button className="button button-outline" disabled={busy} onClick={() => void control("/simulation/reset", { seed: seed === "" ? null : Number(seed) })}>Reset</button></div>
          </div>
        </div>

        <aside className="panel inspector-panel">
          <div className="panel-header"><div><p className="section-label">ENTITY STATE</p><h2>Agent inspector</h2></div><span className="count-badge">{snapshot?.agents.length ?? 0} AGENT</span></div>
          <Inspector agent={selected} reward={snapshot?.cumulative_reward ?? 0} />
          <div className="inspector-foot"><span>WORLD RESOURCES</span><b>{snapshot?.resources.reduce((sum, item) => sum + item.quantity, 0) ?? 0} fiber units</b></div>
        </aside>
      </section>

      <section className="panel metrics-panel">
        <div className="panel-header"><div><p className="section-label">PERFORMANCE</p><h2>Episode metrics</h2></div><span className="metrics-source">REPORTED BY SIMULATION</span></div>
        <TrainingMetrics metrics={snapshot?.metrics ?? []} />
      </section>

      <footer className="footer"><span>ADAPTIVE AGENT FRAMEWORK</span><span>{config ? `Episode limit · ${config.max_episode_steps} steps` : "Connecting to environment"} <i /> Backend snapshots at 10 Hz</span></footer>
    </main>
  );
}
