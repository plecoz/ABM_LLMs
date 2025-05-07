from matplotlib.animation import FuncAnimation
import matplotlib as plt
from .base_plot import BaseMap  # Relative import from same package
from .agent_plot import AgentPlotter


class SimulationAnimator:
    def __init__(self, model, graph):
        self.model = model
        self.graph = graph
        self.base = BaseMap(graph)
        self.plotter = AgentPlotter(self.base)
    
    def update(self, frame):
        """Update plot for each animation frame."""
        self.base.clear_artists()
        self.plotter.plot_pois(
            [a for a in self.model.schedule.agents if hasattr(a, 'poi_type')],
            self.graph
        )
        self.plotter.plot_residents(
            [a for a in self.model.schedule.agents if hasattr(a, 'home_node')],
            self.graph
        )
        self.base.fig.canvas.draw()
    
    def animate(self, steps=10, interval=500):
        """Run the animation."""
        anim = FuncAnimation(
            self.base.fig,
            self.update,
            frames=steps,
            interval=interval,
            repeat=False
        )
        plt.show()