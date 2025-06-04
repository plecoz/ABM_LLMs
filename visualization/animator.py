from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import osmnx as ox
from .base_plot import BaseMap  # Relative import from same package
from .agent_plot import AgentPlotter
import numpy as np
import geopandas as gpd
import os
import networkx as nx


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
        
        # Animation state
        self.current_step = 0
        # Commented out frame interpolation since we now use 1-minute time steps
        # self.frames_per_step = 5  # Number of frames to interpolate between steps
        self.animation = None
        
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
    
    def animate(self, frame):
        """Animation function for FuncAnimation - now updates every step (1 minute)"""
        # Advance the simulation every frame since we no longer use frame interpolation
        self.model.step()
        self.current_step = frame
        
        # Update visualization - no interpolation progress needed
        self.update()
        
        # Check if this is the last frame and print output summary
        if hasattr(self, 'total_frames') and frame >= self.total_frames - 1:
            print("\nAnimation completed!")
            # Print output summary if output controller exists
            if hasattr(self.model, 'output_controller'):
                self.model.output_controller.print_travel_summary()
        
        return self.agent_dots
        
        # Commented out frame interpolation code:
        # # Calculate which simulation step we're on and the progress within that step
        # sim_step = frame // self.frames_per_step
        # progress = (frame % self.frames_per_step) / self.frames_per_step
        # 
        # # Only advance the simulation when starting a new step
        # if frame % self.frames_per_step == 0:
        #     self.model.step()
        #     self.current_step = sim_step
        #     
        # # Update visualization with interpolated positions
        # self.update(progress)
    
    def start_animation(self, num_steps, interval=50):
        """Start the animation loop - now one frame per simulation step"""
        # No longer need to multiply by frames_per_step since we update every step
        total_frames = num_steps
        self.total_frames = total_frames  # Store for completion check
        
        self.animation = FuncAnimation(
            self.fig,
            self.animate,
            frames=total_frames,
            interval=interval,  # Time between frames in milliseconds
            blit=True,
            repeat=False
        )
        
        plt.show()
        
        # Commented out frame interpolation code:
        # total_frames = num_steps * self.frames_per_step
    
    def update(self):
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
    
    def _get_interpolated_position(self, resident, progress):
        """
        Calculate the interpolated position of a traveling resident along the path.
        
        Args:
            resident: The resident agent that is traveling
            progress: Float between 0 and 1 indicating progress within current time step
            
        Returns:
            Tuple (x, y) of the current interpolated position
        """
        # If not traveling or missing necessary attributes, return current position
        if not resident.traveling or not hasattr(resident, 'current_node') or not hasattr(resident, 'destination_node'):
            if hasattr(resident, 'geometry'):
                return resident.geometry.x, resident.geometry.y
            elif hasattr(resident, 'current_node'):
                return self.graph.nodes[resident.current_node]['x'], self.graph.nodes[resident.current_node]['y']
            return 0, 0
            
        try:
            # Get start and end nodes
            start_node = resident.current_node
            end_node = resident.destination_node
            
            # Calculate the shortest path between nodes
            path = nx.shortest_path(self.graph, start_node, end_node, weight='length')
            
            # If path has only one node (start == end), return that position
            if len(path) <= 1:
                return self.graph.nodes[start_node]['x'], self.graph.nodes[start_node]['y']
                
            # Calculate total travel time and remaining time
            total_time = resident.travel_time_remaining + 1  # +1 because we've already decremented by 1
            # Adjust progress to account for both the step progress and the interpolation progress
            continuous_progress = 1 - ((resident.travel_time_remaining - progress) / total_time)
            continuous_progress = max(0, min(1, continuous_progress))  # Clamp between 0 and 1
            
            # Calculate cumulative distances along the path
            distances = [0]  # Start with 0 distance
            total_distance = 0
            
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                # Get edge data
                edge_data = self.graph.get_edge_data(node1, node2, 0)  # 0 is the default key for MultiDiGraph
                # Get length or calculate Euclidean distance
                if 'length' in edge_data:
                    distance = edge_data['length']
                else:
                    x1, y1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y']
                    x2, y2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y']
                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                
                total_distance += distance
                distances.append(total_distance)
            
            # Normalize distances to 0-1 range
            if total_distance > 0:
                distances = [d / total_distance for d in distances]
            
            # Find which segment the agent is on
            target_distance = continuous_progress
            segment_idx = 0
            while segment_idx < len(distances) - 1 and distances[segment_idx + 1] < target_distance:
                segment_idx += 1
                
            # If we're at the last node
            if segment_idx >= len(distances) - 1:
                return self.graph.nodes[path[-1]]['x'], self.graph.nodes[path[-1]]['y']
                
            # Calculate interpolation within the segment
            start_dist = distances[segment_idx]
            end_dist = distances[segment_idx + 1]
            
            if end_dist == start_dist:  # Avoid division by zero
                segment_progress = 0
            else:
                segment_progress = (target_distance - start_dist) / (end_dist - start_dist)
                
            # Get segment start and end positions
            start_x = self.graph.nodes[path[segment_idx]]['x']
            start_y = self.graph.nodes[path[segment_idx]]['y']
            end_x = self.graph.nodes[path[segment_idx + 1]]['x']
            end_y = self.graph.nodes[path[segment_idx + 1]]['y']
            
            # Interpolate position
            x = start_x + segment_progress * (end_x - start_x)
            y = start_y + segment_progress * (end_y - start_y)
            
            return x, y
            
        except Exception as e:
            # If there's any error, return the current position
            print(f"Error calculating interpolated position: {e}")
            if hasattr(resident, 'geometry'):
                return resident.geometry.x, resident.geometry.y
            else:
                # Fallback to current node position
                return self.graph.nodes[resident.current_node]['x'], self.graph.nodes[resident.current_node]['y']
    
    def _plot_residents(self, progress=0.5):
        """Plot all resident agents as colored dots with interpolated positions for traveling agents
        
        Args:
            progress: Progress along the current minute for path interpolation (default 0.5 for mid-step)
        """
        for resident in self.model.residents:
            try:
                if hasattr(resident, 'traveling') and resident.traveling:
                    # Get interpolated position for traveling agents
                    # Use progress to show position along the path within the current travel step
                    x, y = self._get_interpolated_position(resident, progress)
                    
                    # Use a different color for traveling agents
                    #Option to change the color of the traveling agents
                    dot = self.ax.plot(x, y, 'o', color='#00BFFF', markersize=4, markeredgecolor='black',
                                        markeredgewidth=0.5, alpha=0.8)[0]  
                else:
                    # For stationary agents, use the current node position
                    if hasattr(resident, 'current_node'):
                        x, y = self.graph.nodes[resident.current_node]['x'], self.graph.nodes[resident.current_node]['y']
                        dot = self.ax.plot(x, y, 'o', color='#00BFFF', markersize=4,
                                           markeredgecolor='black', markeredgewidth=0.5, alpha=0.8)[0]  # Blue for stationary
                    else:
                        # Use geometry if available
                        x, y = resident.geometry.x, resident.geometry.y
                        dot = self.ax.plot(x, y, 'o', color='#00BFFF', markersize=3, alpha=0.7)[0]
                
                self.agent_dots.append(dot)
            except Exception as e:
                print(f"Error plotting resident {resident.unique_id}: {e}")
    
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
        
        # Add resident agents to legend
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