from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from adaptive_ai.agents.rabbit import Rabbit
from adaptive_ai.environments.rabbit_survival.world import World


class RabbitRenderer:
    def __init__(self, mode: str, fps: int) -> None:
        self.mode = mode
        self.fps = fps
        if mode == "human":
            self.figure, self.axis = plt.subplots(figsize=(6, 5))
            plt.show(block=False)
        else:
            self.figure = Figure(figsize=(6, 5))
            FigureCanvasAgg(self.figure)
            self.axis = self.figure.subplots()

    def draw(
        self, world: World, rabbit: Rabbit, action: str, cumulative_reward: float,
    ) -> NDArray[np.uint8] | None:
        axis = self.axis
        axis.clear()
        axis.set(xlim=(-0.5, world.config.width - 0.5),
                 ylim=(world.config.height - 0.5, -0.5), aspect="equal")
        axis.set_xticks(range(world.config.width))
        axis.set_yticks(range(world.config.height))
        axis.grid(alpha=0.3)
        if world.plants:
            xs, ys = zip(*world.plants)
            axis.scatter(xs, ys, color="forestgreen", marker="s", label="Plant Fiber")
        x, y = rabbit.position
        axis.scatter(x, y, color="royalblue" if rabbit.alive else "gray", label="Rabbit")
        axis.legend(loc="upper right")
        axis.set_title(
            f"Step {rabbit.age} | Action: {action} | Reward: {cumulative_reward:.2f}\n"
            f"Energy {rabbit.energy:.1f}/{rabbit.max_energy:g} | "
            f"Fatigue {rabbit.fatigue:.1f}/{rabbit.max_fatigue:g} | "
            f"Inventory {len(rabbit.inventory)}/{rabbit.inventory_capacity}"
        )
        canvas = self.figure.canvas
        if self.mode == "human":
            if plt.fignum_exists(self.figure.number):
                canvas.draw_idle()
                canvas.start_event_loop(1 / self.fps)
            return None
        assert isinstance(canvas, FigureCanvasAgg)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

    def close(self) -> None:
        if self.mode == "human":
            plt.close(self.figure)
