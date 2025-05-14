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
            dot = self.ax.plot(x, y, 'o', color='#00BFFF', markersize=8, alpha=0.7)[0]  # Bright deep sky blue
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
        print(f"Total organizations in model: {len(self.model.organizations)}")
        for i, organization in enumerate([o for o in self.model.organizations if hasattr(o, 'current_node')]):
            print(f"Organization {i}: ID={organization.unique_id}, current_node={organization.current_node}, org_type={organization.org_type}")
            if organization.current_node is not None:
                try:
                    x, y = self.graph.nodes[organization.current_node]['x'], self.graph.nodes[organization.current_node]['y']
                    
                    # Always use black for organizations, with different marker shapes for types
                    color = '#000000'  # Black for all organizations
                    
                    # Use different marker shapes based on organization type
                    if hasattr(organization, 'org_type'):
                        if organization.org_type == 'school':
                            marker = 's'  # square
                        elif organization.org_type == 'hospital':
                            marker = 'h'  # hexagon
                        elif organization.org_type == 'business':
                            marker = 'd'  # diamond
                        else:
                            marker = 'o'  # circle
                    else:
                        marker = 'o'  # circle
                    
                    # Make organizations extremely visible
                    dot = self.ax.plot(x, y, marker, color=color, markersize=20, alpha=1.0, 
                                      markeredgewidth=2, markeredgecolor='white')[0]
                    self.agent_dots.append(dot)
                    print(f"  - Successfully plotted at ({x}, {y})")
                    
                    # Add a text label
                    self.ax.text(x, y, f"Org {organization.unique_id}", fontsize=10, 
                                ha='center', va='bottom', color='white',
                                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))
                    
                except Exception as e:
                    print(f"  - Error plotting organization {organization.unique_id}: {e}")
    
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
            mpatches.Patch(facecolor='#00BFFF', edgecolor='black', label='Residents'),
            mpatches.Patch(facecolor='#000000', edgecolor='black', label='Organizations')
        ]
        
        # Organization type markers
        org_markers = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#000000', 
                      markersize=10, label='School'),
            plt.Line2D([0], [0], marker='h', color='w', markerfacecolor='#000000', 
                      markersize=10, label='Hospital'),
            plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='#000000', 
                      markersize=10, label='Business'),
        ]
        
        # Create the legend
        legend1 = self.ax.legend(handles=poi_legend_elements, 
                               loc='upper left', 
                               title="Points of Interest",
                               framealpha=0.8,
                               bbox_to_anchor=(1.01, 1))
        
        # Add the second legend
        self.ax.add_artist(legend1)
        legend2 = self.ax.legend(handles=agent_legend_elements, 
                     loc='upper left', 
                     title="Agents",
                     framealpha=0.8,
                     bbox_to_anchor=(1.01, 0.6))
        
        # Add the third legend for organization types
        self.ax.add_artist(legend2)
        self.ax.legend(handles=org_markers,
                     loc='upper left',
                     title="Organization Types",
                     framealpha=0.8,
                     bbox_to_anchor=(1.01, 0.3))
        
        # Adjust figure to make room for legend
        self.fig.tight_layout()
        self.fig.subplots_adjust(right=0.75)