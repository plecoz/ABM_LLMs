from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import osmnx as ox
from .base_plot import BaseMap  # Relative import from same package
from .agent_plot import AgentPlotter


class SimulationAnimator:
    def __init__(self, model, graph, ax=None):
        self.model = model
        self.graph = graph
        self.fig = ax.figure if ax else plt.figure(figsize=(12, 10))
        self.ax = ax if ax else self.fig.add_subplot(111)
        
        # Initialize plot elements
        self.base_plot = self._create_base_plot()
        self.agent_dots = []
        self.poi_markers = []
    
    def _create_base_plot(self):
        """Draw the static map background"""
        return ox.plot_graph(
            self.graph,
            ax=self.ax,
            show=False,
            close=False,
            bgcolor="#f5f5f5",
            node_size=0,
            edge_color="lightgray"
        )
    
    def initialize(self):
        """Draw initial state"""
        self._plot_pois()
        self._plot_residents()
        self._plot_organizations()
        plt.draw()
    
    def update(self, step):
        """Update dynamic elements"""
        # Clear previous agents
        for dot in self.agent_dots:
            dot.remove()
        self.agent_dots = []
        
        # Redraw agents
        self._plot_residents()
        self._plot_organizations()
        
        # Refresh display
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def _plot_residents(self):
        """Plot all agents as colored dots"""
        for resident in [r for r in self.model.residents if hasattr(r, 'current_node')]:
            x, y = self.graph.nodes[resident.current_node]['x'], self.graph.nodes[resident.current_node]['y']
            dot = self.ax.plot(x, y, 'o', color='blue', markersize=2)[0]
            self.agent_dots.append(dot)
    
    def _plot_pois(self):
        """Plot all POIs (only called once)"""
        for poi_type, nodes in self.model.pois.items():
            for node in nodes:
                x, y = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']
                marker = self.ax.plot(x, y, 's', color='red', markersize=3)[0]
                self.poi_markers.append(marker)
    
    def _plot_organizations(self):
        """Plot all organizations as colored dots (called once, organizations don't move)"""
        for organization in [o for o in self.model.organizations if hasattr(o, 'current_node')]:
            if organization.current_node is not None:
                x, y = self.graph.nodes[organization.current_node]['x'], self.graph.nodes[organization.current_node]['y']
                dot = self.ax.plot(x, y, 's', color='lime', markersize=6)[0]
                self.agent_dots.append(dot)