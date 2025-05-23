from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import osmnx as ox
from .base_plot import BaseMap  # Relative import from same package
from .agent_plot import AgentPlotter
import numpy as np
import geopandas as gpd
import os


class SimulationAnimator:
    def __init__(self, model, graph, ax=None, parishes_gdf=None):
        self.model = model
        self.graph = graph
        self.fig = ax.figure if ax else plt.figure(figsize=(12, 10))
        self.ax = ax if ax else self.fig.add_subplot(111)
        self.parishes_gdf = parishes_gdf  # GeoDataFrame containing parishes
        
        # Initialize plot elements
        self.base_plot = self._create_base_plot()
        self.agent_dots = []
        self.poi_markers = []
        
        # Define color schemes for POI categories
        self.poi_colors = {
            "daily_living": "#FF9800",   # Orange
            "healthcare": "#F44336",     # Red
            "education": "#2196F3",      # Blue
            "entertainment": "#4CAF50",  # Green
            "transportation": "#000000", # Black
            "other": "#9E9E9E"           # Gray
        }
        
        # Define symbols for POI categories
        self.poi_symbols = {
            "daily_living": "s",         # Square
            "healthcare": "h",           # Hexagon
            "education": "^",            # Triangle up
            "entertainment": "*",        # Star
            "transportation": "d",       # Diamond
            "other": "o"                 # Circle
        }
        
        # Flag to determine whether to use specific POI styling
        self.use_specific_poi_styling = True
    
    def _create_base_plot(self):
        """Draw the static map background with parishes if available"""
        # First plot parishes if available
        if self.parishes_gdf is not None:
            self.parishes_gdf.plot(
                ax=self.ax,
                column='name',
                cmap='tab20',
                alpha=0.5,
                edgecolor='black',
                legend=True,
                legend_kwds={'title': 'Parishes', 'loc': 'lower left'}
            )
            
            # Add parish labels
            for x, y, label in zip(
                self.parishes_gdf.geometry.centroid.x,
                self.parishes_gdf.geometry.centroid.y,
                self.parishes_gdf['name']
            ):
                self.ax.text(x, y, label, fontsize=8, ha='center')
        
        # Then plot the street network on top
        return ox.plot_graph(
            self.graph,
            ax=self.ax,
            show=False,
            close=False,
            bgcolor="#f5f5f5" if self.parishes_gdf is None else None,  # Only use bgcolor if no parishes
            node_size=0,
            edge_color="gray",
            edge_linewidth=0.7
        )
    
    def initialize(self):
        """Draw initial state"""
        self._plot_poi_agents()
        self._plot_residents()
        self._create_legend()
        
        # Set title with parishes information
        if self.parishes_gdf is not None:
            self.ax.set_title("Macau 15-Minute City with Parishes", fontsize=16)
        else:
            self.ax.set_title("Macau 15-Minute City", fontsize=16)
            
        plt.draw()
    
    def update(self, step):
        """Update dynamic elements"""
        # Clear previous agents
        for dot in self.agent_dots:
            dot.remove()
        self.agent_dots = []
        
        # Redraw agents
        self._plot_residents()
        
        # Refresh display
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def _plot_residents(self):
        """Plot all resident agents as colored dots"""
        for resident in [r for r in self.model.residents if hasattr(r, 'current_node')]:
            x, y = self.graph.nodes[resident.current_node]['x'], self.graph.nodes[resident.current_node]['y']
            dot = self.ax.plot(x, y, 'o', color='#00BFFF', markersize=3, alpha=0.7)[0]  # Bright deep sky blue
            self.agent_dots.append(dot)
    
    def _plot_poi_agents(self):
        """Plot all POI agents from the model's poi_agents list"""
        # Track how many POIs of each category we plot
        category_counts = {
            "daily_living": 0,
            "healthcare": 0,
            "education": 0,
            "entertainment": 0,
            "transportation": 0,
            "other": 0
        }
        
        for poi in self.model.poi_agents:
            try:
                x, y = poi.geometry.x, poi.geometry.y
                
                # Get category and appropriate styling
                category = poi.category if hasattr(poi, 'category') else 'other'
                
                # Count this POI
                if category in category_counts:
                    category_counts[category] += 1
                else:
                    category_counts['other'] += 1
                
                # Special case for transportation (bus stops)
                if category == 'transportation':
                    color = '#000000'  # Black
                    symbol = 'd'       # Diamond
                    size = 4           # Small size
                else:
                    color = self.poi_colors.get(category, "#9E9E9E")  # Default to gray
                    symbol = self.poi_symbols.get(category, "o")      # Default to circle
                    size = 6           # Standard size
                
                # Create marker with appropriate styling
                marker = self.ax.plot(x, y, symbol, color=color,
                                     markersize=size, alpha=0.8)[0]
                
                self.poi_markers.append(marker)
            except Exception as e:
                print(f"Error plotting POI agent {poi.unique_id}: {e}")
        
        # Print summary of what was actually plotted
        print("POIs plotted by category:")
        for category, count in category_counts.items():
            if count > 0:
                print(f"  - {category}: {count}")
        print("")
    
    def _create_legend(self):
        """Create a legend for the visualization"""
        legend_elements = []
        
        # Add POI category elements to legend
        legend_titles = {
            "daily_living": "Daily Living",
            "healthcare": "Healthcare",
            "education": "Education",
            "entertainment": "Entertainment",
            "transportation": "Bus Stops",
            "other": "Other POIs"
        }
        
        for category, color in self.poi_colors.items():
            symbol = self.poi_symbols.get(category, 'o')
            size = 4 if category == 'transportation' else 6
            
            legend_elements.append(
                plt.Line2D([0], [0], marker=symbol, color='w',
                           markerfacecolor=color, markersize=8,
                           label=legend_titles.get(category, category.capitalize()))
            )
        
        # Add resident agent to legend
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#00BFFF', markersize=8,
                      label='Residents')
        )
        
        # Create the legend
        self.ax.legend(handles=legend_elements, 
                      loc='upper right',
                      title="Map Elements",
                      framealpha=0.8)
        
        # Adjust figure to make room for legend
        self.fig.tight_layout()