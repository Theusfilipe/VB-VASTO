"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { MouseEvent } from "react";
import type { AgentSnapshot, ResourceSnapshot } from "@/lib/types";

type Props = {
  width: number;
  height: number;
  agents: AgentSnapshot[];
  resources: ResourceSnapshot[];
  selectedAgent: string | null;
  onSelectAgent: (identifier: string) => void;
};

const INSET = 14;

export function WorldCanvas({
  width,
  height,
  agents,
  resources,
  selectedAgent,
  onSelectAgent,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasWidth, setCanvasWidth] = useState(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvasWidth === 0) return;
    const ratio = window.devicePixelRatio || 1;
    const cssHeight = Math.max(240, canvasWidth * (height / width));
    canvas.style.height = `${cssHeight}px`;
    canvas.width = Math.round(canvasWidth * ratio);
    canvas.height = Math.round(cssHeight * ratio);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    context.clearRect(0, 0, canvasWidth, cssHeight);
    const cell = Math.min(
      (canvasWidth - INSET * 2) / width,
      (cssHeight - INSET * 2) / height,
    );
    const left = (canvasWidth - cell * width) / 2;
    const top = (cssHeight - cell * height) / 2;

    context.fillStyle = "#fbfcfa";
    context.fillRect(left, top, cell * width, cell * height);
    context.strokeStyle = "#e3e9e3";
    context.lineWidth = 1;
    for (let x = 0; x <= width; x += 1) {
      context.beginPath();
      context.moveTo(left + x * cell, top);
      context.lineTo(left + x * cell, top + height * cell);
      context.stroke();
    }
    for (let y = 0; y <= height; y += 1) {
      context.beginPath();
      context.moveTo(left, top + y * cell);
      context.lineTo(left + width * cell, top + y * cell);
      context.stroke();
    }

    for (const resource of resources) {
      const x = left + (resource.position.x + 0.5) * cell;
      const y = top + (resource.position.y + 0.5) * cell;
      context.fillStyle = "#75a684";
      context.beginPath();
      context.arc(x, y, Math.max(5, cell * 0.13), 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#fff";
      context.font = `600 ${Math.max(9, cell * 0.16)}px ui-sans-serif`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(String(resource.quantity), x, y);
    }

    for (const agent of agents) {
      const x = left + (agent.position.x + 0.5) * cell;
      const y = top + (agent.position.y + 0.5) * cell;
      if (agent.identifier === selectedAgent && agent.perception_radius > 0) {
        context.fillStyle = "rgba(102, 139, 188, 0.10)";
        context.strokeStyle = "rgba(102, 139, 188, 0.45)";
        context.lineWidth = 1.5;
        context.beginPath();
        context.arc(x, y, agent.perception_radius * cell, 0, Math.PI * 2);
        context.fill();
        context.stroke();
      }
      context.fillStyle = agent.alive ? "#355d86" : "#a8aaa4";
      context.strokeStyle = "#fff";
      context.lineWidth = 3;
      context.beginPath();
      context.arc(x, y, cell * 0.2, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.fillStyle = "#fff";
      context.font = `600 ${cell * 0.22}px ui-sans-serif`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText("R", x, y + 1);
    }
    context.strokeStyle = "#aebbb0";
    context.lineWidth = 2;
    context.strokeRect(left, top, cell * width, cell * height);
  }, [agents, canvasWidth, height, resources, selectedAgent, width]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(([entry]) => {
      setCanvasWidth(entry.contentRect.width);
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  useEffect(() => draw(), [draw]);

  function selectAt(event: MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas || agents.length === 0) return;
    const bounds = canvas.getBoundingClientRect();
    const px = event.clientX - bounds.left;
    const py = event.clientY - bounds.top;
    const cell = Math.min(
      (bounds.width - INSET * 2) / width,
      (bounds.height - INSET * 2) / height,
    );
    const left = (bounds.width - cell * width) / 2;
    const top = (bounds.height - cell * height) / 2;
    const nearest = agents
      .map((agent) => ({
        agent,
        distance: Math.hypot(
          px - (left + (agent.position.x + 0.5) * cell),
          py - (top + (agent.position.y + 0.5) * cell),
        ),
      }))
      .filter(({ distance }) => distance <= cell * 0.35)
      .sort((a, b) => a.distance - b.distance)[0];
    if (nearest) onSelectAgent(nearest.agent.identifier);
  }

  return (
    <canvas
      ref={canvasRef}
      onClick={selectAt}
      aria-label="Simulation world. Click an agent to inspect it."
      className="world-canvas"
    />
  );
}
