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
        
        # Define specific colors and symbols for selected POI types
        self.specific_poi_colors = {
            "hospital": "#FF0000",      # Bright red
            "school": "#0000FF",        # Bright blue
            "bank": "#FFD700",          # Gold
            "police": "#800080",        # Purple
            "fire_station": "#FF4500",  # Orange-red
        }
        
        self.specific_poi_symbols = {
            "hospital": "H",            # Hospital symbol
            "school": "s",              # Square
            "bank": "^",                # Dollar sign
            "police": "P",              # P for police
            "fire_station": "X",        # X for fire station
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
        self._plot_pois()
        self._plot_residents()
        self._plot_organizations()
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
        self._plot_organizations()
        
        # Refresh display
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def _plot_residents(self):
        """Plot all agents as colored dots"""
        for resident in [r for r in self.model.residents if hasattr(r, 'current_node')]:
            x, y = self.graph.nodes[resident.current_node]['x'], self.graph.nodes[resident.current_node]['y']
            dot = self.ax.plot(x, y, 'o', color='#00BFFF', markersize=3, alpha=0.7)[0]  # Bright deep sky blue
            self.agent_dots.append(dot)
    
    def _plot_pois(self):
        """Plot all POIs with category-specific colors and shapes"""
        for category, poi_list in self.model.pois.items():
            for poi_entry in poi_list:
                if isinstance(poi_entry, tuple):
                    node, poi_type = poi_entry  # Unpack node and type
                else:
                    node = poi_entry  # If it's just a node ID
                    poi_type = category
                
                try:
                    x, y = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']
                    
                    # Use specific styling for selected POI types if enabled
                    if self.use_specific_poi_styling and poi_type in self.specific_poi_colors:
                        color = self.specific_poi_colors[poi_type]
                        symbol = self.specific_poi_symbols[poi_type]
                        
                        # Make selected POIs more prominent
                        marker = self.ax.plot(x, y, symbol, color=color, 
                                             markersize=12, markeredgecolor='black',
                                             markeredgewidth=1.0, alpha=0.9)[0]
                        
                        # Add a text label for important POIs
                        #self.ax.text(x, y + 0.0001, f"{poi_type}", fontsize=8, 
                        #            ha='center', va='bottom', color='black',
                        #            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
                    else:
                        # Use category-based styling for other POIs
                        color = self.poi_colors.get(category, "#000000")
                        symbol = self.poi_symbols.get(category, "o")
                        
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
            #print(f"Organization {i}: ID={organization.unique_id}, current_node={organization.current_node}, org_type={organization.org_type}")
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
        legend_elements = []
        
        # Add specific POI types to legend if enabled
        if self.use_specific_poi_styling:
            for poi_type, color in self.specific_poi_colors.items():
                symbol = self.specific_poi_symbols.get(poi_type, 'o')
                legend_elements.append(
                    plt.Line2D([0], [0], marker=symbol, color='w', 
                              markerfacecolor=color, markersize=10,
                              markeredgecolor='black', markeredgewidth=1.0,
                              label=poi_type.replace('_', ' ').title())
                )
        
        # Add category-based POIs if any aren't covered by specific styling
        if not self.use_specific_poi_styling:
            for category, color in self.poi_colors.items():
                symbol = self.poi_symbols.get(category, 'o')
                legend_elements.append(
                    plt.Line2D([0], [0], marker=symbol, color='w',
                              markerfacecolor=color, markersize=10,
                              label=category.capitalize())
                )
        
        # Agent legends
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#00BFFF', markersize=10,
                      label='Residents')
        )
        
        # Organization type markers
        org_types = {
            'School': 's',
            'Hospital': 'h',
            'Business': 'd',
            'Other': 'o'
        }
        
        for org_type, marker in org_types.items():
            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, color='w',
                          markerfacecolor='#000000', markersize=10,
                          markeredgecolor='white', markeredgewidth=1.0,
                          label=org_type)
            )
        
        # Create the legend
        self.ax.legend(handles=legend_elements, 
                     loc='upper left', 
                     title="Map Elements",
                     framealpha=0.8,
                     bbox_to_anchor=(1.01, 1))
        
        # Adjust figure to make room for legend
        self.fig.tight_layout()
        self.fig.subplots_adjust(right=0.75)
        
    def set_poi_styling(self, use_specific_styling=True):
        """
        Set whether to use specific POI styling.
        
        Args:
            use_specific_styling: If True, use specific styling for selected POI types.
                                 If False, use category-based styling.
        """
        self.use_specific_poi_styling = use_specific_styling