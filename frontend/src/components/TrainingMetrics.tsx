"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EpisodeMetric } from "@/lib/types";

const charts = [
  { key: "reward", title: "Episode reward", color: "#4b7399" },
  { key: "survival_duration", title: "Survival duration", color: "#668b68" },
  { key: "mean_survival_duration", title: "Recent mean survival", color: "#9b8052" },
  { key: "food_consumption", title: "Food consumption", color: "#728f6b" },
  { key: "sleeping_frequency", title: "Sleeping frequency", color: "#897ba5" },
] as const;

export function TrainingMetrics({ metrics }: { metrics: EpisodeMetric[] }) {
  if (metrics.length === 0) {
    return (
      <div className="empty-chart">
        <span className="empty-chart-mark">⌁</span>
        <p>Episode metrics will appear after an episode completes.</p>
        <small>No training or evaluation results have been reported yet.</small>
      </div>
    );
  }

  return (
    <div className="metrics-grid">
      {charts.map((chart) => (
        <section className="metric-chart" key={chart.key}>
          <div className="chart-heading">
            <h3>{chart.title}</h3>
            <span>{metrics.length} episodes</span>
          </div>
          <ResponsiveContainer width="100%" height={144}>
            <LineChart data={metrics} margin={{ top: 8, right: 12, bottom: 0, left: -18 }}>
              <CartesianGrid stroke="#edf0ec" strokeDasharray="3 4" vertical={false} />
              <XAxis dataKey="episode" tickLine={false} axisLine={false} tick={{ fontSize: 10, fill: "#829087" }} />
              <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 10, fill: "#829087" }} />
              <Tooltip
                contentStyle={{ border: "1px solid #e8ece7", borderRadius: 8, fontSize: 12 }}
                formatter={(value) => [Number(value).toFixed(2), chart.title]}
                labelFormatter={(label) => `Episode ${label}`}
              />
              <Line
                type="monotone"
                dataKey={chart.key}
                stroke={chart.color}
                strokeWidth={2}
                dot={{ r: 2, fill: chart.color, strokeWidth: 0 }}
                activeDot={{ r: 4 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </section>
      ))}
    </div>
  );
}
