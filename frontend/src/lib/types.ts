export type Position = { x: number; y: number };

export type AgentSnapshot = {
  identifier: string;
  position: Position;
  energy: number;
  max_energy: number;
  fatigue: number;
  max_fatigue: number;
  inventory: number;
  inventory_capacity: number;
  current_action: string;
  perception_radius: number;
  age: number;
  alive: boolean;
};

export type ResourceSnapshot = {
  kind: string;
  position: Position;
  quantity: number;
};

export type EpisodeMetric = {
  episode: number;
  reward: number;
  survival_duration: number;
  mean_survival_duration: number;
  food_consumption: number;
  sleeping_frequency: number;
};

export type SimulationSnapshot = {
  dimensions: { width: number; height: number };
  simulation_step: number;
  episode: number;
  status: "running" | "paused";
  execution_mode: "simulation" | "evaluation" | "training";
  speed: 1 | 2 | 5 | 10;
  resources: ResourceSnapshot[];
  agents: AgentSnapshot[];
  cumulative_reward: number;
  metrics: EpisodeMetric[];
  policy_name: string | null;
  training_run: string | null;
  training_timesteps: number | null;
};

export type SimulationConfig = {
  width: number;
  height: number;
  max_episode_steps: number;
  resource_density: number;
  regeneration_probability: number;
  perception_radius: number;
};

export type PolicyCatalog = {
  configured: boolean;
  policies: string[];
};

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export const WS_URL = API_URL.replace(/^http/, "ws") + "/ws/simulation";
