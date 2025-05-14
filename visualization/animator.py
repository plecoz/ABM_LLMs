from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import osmnx as ox
from .base_plot import BaseMap  # Relative import from same package
from .agent_plot import AgentPlotter
import numpy as np


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
        
        # Define color schemes for POIs
        self.poi_colors = {
            "healthcare": "#D32F2F",    # Red
            "education": "#1976D2",     # Blue
            "shopping": "#FFC107",      # Amber 
            "recreation": "#388E3C",    # Green
            "services": "#7B1FA2",      # Purple
            "food": "#FF5722"           # Deep Orange
        }
        
        # Define symbols for POIs
        self.poi_symbols = {
            "healthcare": "H",          # Hospital symbol
            "education": "s",           # Square for education
            "shopping": "P",            # Plus for shopping
            "recreation": "^",          # Triangle for recreation
            "services": "D",            # Diamond for services
            "food": "*"                 # Star for food
        }
    
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
        self._create_legend()
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
            dot = self.ax.plot(x, y, 'o', color='#2196F3', markersize=4, alpha=0.7)[0]
            self.agent_dots.append(dot)
    
    def _plot_pois(self):
        """Plot all POIs with category-specific colors and shapes"""
        for category, poi_list in self.model.pois.items():
            color = self.poi_colors.get(category, "#000000")
            symbol = self.poi_symbols.get(category, "o")
            
            for poi_entry in poi_list:
                if isinstance(poi_entry, tuple):
                    node, _ = poi_entry  # Unpack node and type
                else:
                    node = poi_entry  # If it's just a node ID
                
                try:
                    x, y = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']
                    marker = self.ax.plot(x, y, symbol, color=color, 
                                         markersize=8, markeredgecolor='black',
                                         markeredgewidth=0.5)[0]
                    self.poi_markers.append(marker)
                except (KeyError, TypeError) as e:
                    print(f"Error plotting POI at node {node}: {e}")
    
    def _plot_organizations(self):
        """Plot all organizations as colored dots"""
        for organization in [o for o in self.model.organizations if hasattr(o, 'current_node')]:
            if organization.current_node is not None:
                x, y = self.graph.nodes[organization.current_node]['x'], self.graph.nodes[organization.current_node]['y']
                
                # Use different color based on organization type
                if hasattr(organization, 'org_type'):
                    if organization.org_type == 'school':
                        color = self.poi_colors['education']
                    elif organization.org_type == 'hospital':
                        color = self.poi_colors['healthcare']
                    elif organization.org_type == 'business':
                        color = self.poi_colors['shopping']
                    else:
                        color = '#009688'  # Default teal
                else:
                    color = '#009688'  # Default teal
                    
                dot = self.ax.plot(x, y, 'o', color=color, markersize=8, alpha=0.9)[0]
                self.agent_dots.append(dot)
    
    def _create_legend(self):
        """Create a legend for the visualization"""
        # POI category legends
        poi_legend_elements = []
        for category, color in self.poi_colors.items():
            symbol = self.poi_symbols.get(category, 'o')
            # Create patch for legend
            poi_legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='black',
                               label=f"{category.capitalize()}")
            )
        
        # Agent legends
        agent_legend_elements = [
            mpatches.Patch(facecolor='#2196F3', edgecolor='black', label='Residents'),
            mpatches.Patch(facecolor='#009688', edgecolor='black', label='Organizations')
        ]
        
        # Create the legend
        legend1 = self.ax.legend(handles=poi_legend_elements, 
                               loc='upper left', 
                               title="Points of Interest",
                               framealpha=0.8,
                               bbox_to_anchor=(1.01, 1))
        
        # Add the second legend
        self.ax.add_artist(legend1)
        self.ax.legend(handles=agent_legend_elements, 
                     loc='upper left', 
                     title="Agents",
                     framealpha=0.8,
                     bbox_to_anchor=(1.01, 0.6))
        
        # Adjust figure to make room for legend
        self.fig.tight_layout()
        self.fig.subplots_adjust(right=0.85)