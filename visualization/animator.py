from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import patheffects
import osmnx as ox
from .base_plot import BaseMap  # Relative import from same package

from .interactive_matplotlib import InteractiveEnhancer, Enhanced3DBuildings
import numpy as np
import geopandas as gpd
import os
import networkx as nx
from matplotlib_scalebar.scalebar import ScaleBar


class SimulationAnimator:
    def __init__(self, model, graph, ax=None, parishes_gdf=None, residential_buildings=None, water_bodies=None, cliffs=None, forests=None, buildings_3d=None, interactive=False):
        self.model = model
        self.graph = graph
        self.fig = ax.figure if ax else plt.figure(figsize=(12, 10))
        self.ax = ax if ax else self.fig.add_subplot(111)
        self.parishes_gdf = parishes_gdf  # GeoDataFrame containing parishes
        self.residential_buildings = residential_buildings
        self.water_bodies = water_bodies  # GeoDataFrame containing water bodies
        self.cliffs = cliffs  # GeoDataFrame containing cliffs and barriers
        self.forests = forests  # GeoDataFrame containing forests and green areas
        self.buildings_3d = buildings_3d  # GeoDataFrame containing 3D buildings
        
        # Interactive features
        self.interactive = interactive
        self.interactive_enhancer = None
        self.building_plotter = None
        
        # Set font to support Chinese characters if available
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign is displayed correctly
        except Exception:
            print("Warning: CJK-compatible font not found. Chinese characters in plot may not display correctly.")
        
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
            "casino": "#9C27B0",         # Purple
            "other": "#9E9E9E"           # Gray
        }
        
        # Define symbols for POI categories
        self.poi_symbols = {
            "daily_living": "s",         # Square
            "healthcare": "h",           # Hexagon
            "education": "^",            # Triangle up
            "entertainment": "*",        # Star
            "transportation": "d",       # Diamond
            "casino": "D",               # Large diamond
            "other": "o"                 # Circle
        }
        
        # Flag to determine whether to use specific POI styling
        self.use_specific_poi_styling = True
        
        # Setup interactive features if requested
        if self.interactive:
            self._setup_interactive_features()
    
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
            
            # For accurate centroid calculation, project to a local UTM zone
            try:
                # Estimate the UTM CRS for the geometries, which is more robust
                utm_crs = self.parishes_gdf.estimate_utm_crs()
                # Project to the estimated UTM CRS
                projected_parishes = self.parishes_gdf.to_crs(utm_crs)
                
                # Calculate centroids in projected space
                projected_centroids = projected_parishes.geometry.centroid
                # Convert centroids back to the original CRS for plotting
                centroids_geo = projected_centroids.to_crs(self.parishes_gdf.crs)

                # Add parish labels at the accurate centroid
                for geom, label in zip(centroids_geo, self.parishes_gdf['name']):
                    self.ax.text(geom.x, geom.y, label, fontsize=8, ha='center',
                                 path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
            except Exception as e:
                print(f"Warning: Could not plot parish labels due to projection error: {e}")
        
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
    
    def _setup_interactive_features(self):
        """Setup interactive features for the plot."""
        print("Setting up interactive features...")
        
        # Create interactive enhancer
        self.interactive_enhancer = InteractiveEnhancer(self.ax, self.fig)
        
        # Create building plotter if 3D buildings are available
        if self.buildings_3d is not None:
            self.building_plotter = Enhanced3DBuildings(self.ax)
            
        # Add layer controls
        self.interactive_enhancer.add_layer_controls()
        
        # Update figure layout for controls
        self.fig.subplots_adjust(bottom=0.1)
        
        # Add instructions
        instructions = "🖱️ Scroll: Zoom | Drag: Pan | R: Reset | H: Help | Left Panel: Toggle Layers"
        self.fig.text(0.5, 0.02, instructions, ha='center', fontsize=10, 
                     style='italic', bbox=dict(boxstyle="round,pad=0.3", 
                     facecolor="lightblue", alpha=0.7))
        
        print("✅ Interactive features ready!")
    
    def initialize(self):
        """Initialize the plot with the static elements."""
        self.ax.clear()
        
        # Plot water bodies first (lowest layer)
        if self.water_bodies is not None and not self.water_bodies.empty:
            collections_before = len(self.ax.collections)
            water_artist = self.water_bodies.plot(ax=self.ax, facecolor='#4FC3F7', edgecolor='#0277BD', linewidth=0.5, alpha=0.7, zorder=0)
            if self.interactive_enhancer:
                # Register the new collections that were added
                for collection in self.ax.collections[collections_before:]:
                    self.interactive_enhancer.register_layer_artist(collection, 'streets')
        
        # Plot forests and green areas
        if self.forests is not None and not self.forests.empty:
            collections_before = len(self.ax.collections)
            forest_artist = self.forests.plot(ax=self.ax, facecolor='#66BB6A', edgecolor='#2E7D32', linewidth=0.5, alpha=0.6, zorder=1)
            if self.interactive_enhancer:
                # Register the new collections that were added
                for collection in self.ax.collections[collections_before:]:
                    self.interactive_enhancer.register_layer_artist(collection, 'streets')
        
        # Plot cliffs and barriers
        if self.cliffs is not None and not self.cliffs.empty:
            collections_before = len(self.ax.collections)
            cliff_artist = self.cliffs.plot(ax=self.ax, facecolor='#8D6E63', edgecolor='#5D4037', linewidth=0.8, alpha=0.8, zorder=2)
            if self.interactive_enhancer:
                # Register the new collections that were added
                for collection in self.ax.collections[collections_before:]:
                    self.interactive_enhancer.register_layer_artist(collection, 'streets')
        
        # Plot residential buildings
        if self.residential_buildings is not None and not self.residential_buildings.empty:
            collections_before = len(self.ax.collections)
            residential_artist = self.residential_buildings.plot(ax=self.ax, facecolor='#d3d3d3', edgecolor='gray', linewidth=0.5, zorder=3)
            if self.interactive_enhancer:
                # Register the new collections that were added
                for collection in self.ax.collections[collections_before:]:
                    self.interactive_enhancer.register_layer_artist(collection, 'buildings')
        
        # Plot 3D buildings with enhanced visualization
        if self.buildings_3d is not None and not self.buildings_3d.empty and self.building_plotter:
            self.building_plotter.plot_3d_buildings(self.buildings_3d, self.interactive_enhancer)
            self.building_plotter.add_building_info_on_click(self.buildings_3d)
        
        # Plot the street network
        lines_before = len(self.ax.lines)
        street_plot = ox.plot_graph(self.graph, ax=self.ax, node_size=0, edge_color='gray', edge_linewidth=0.5, show=False, close=False)
        if self.interactive_enhancer:
            # Register the new lines that were added
            for line in self.ax.lines[lines_before:]:
                self.interactive_enhancer.register_layer_artist(line, 'streets')
        
        # Plot parishes if available (on top of everything else)
        if self.parishes_gdf is not None:
            collections_before = len(self.ax.collections)
            parish_artist = self.parishes_gdf.plot(ax=self.ax, edgecolor='black', facecolor='none', linewidth=1.5, zorder=5)
            if self.interactive_enhancer:
                # Register the new collections that were added
                for collection in self.ax.collections[collections_before:]:
                    self.interactive_enhancer.register_layer_artist(collection, 'streets')

        # Plot POIs
        self._plot_poi_agents()
        
        # Plot initial resident positions
        self._plot_residents()
        
        self._add_scale_bar()
        self._add_north_arrow()
        self._create_legend()
        
        # Set title with environment information
        title_parts = ["Macau 15-Minute City"]
        if self.interactive:
            title_parts.append("(Interactive)")
        if self.parishes_gdf is not None:
            title_parts.append("with Parishes")
        if (self.water_bodies is not None and not self.water_bodies.empty) or (self.cliffs is not None and not self.cliffs.empty) or (self.forests is not None and not self.forests.empty):
            title_parts.append("and Environment")
        if self.buildings_3d is not None:
            title_parts.append("+ 3D Buildings")
        
        self.ax.set_title(" ".join(title_parts), fontsize=16)
        
        # Re-setup interactive features after clearing
        if self.interactive and self.interactive_enhancer:
            self.interactive_enhancer._setup_interaction()
            
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
        Now works with 80-meter steps where each step = 80 meters of travel.
        
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
                
            # Calculate total path distance
            total_distance = 0
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                edge_data = self.graph.get_edge_data(node1, node2, 0)
                if 'length' in edge_data:
                    total_distance += edge_data['length']
                else:
                    x1, y1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y']
                    x2, y2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y']
                    total_distance += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            
            # Calculate how far the agent should be along the path
            # We need to know the original travel time to calculate progress correctly
            # Since we don't store the original travel time, we'll calculate it from the path
            original_travel_time = max(1, int(np.ceil(total_distance / 80.0)))
            
            # Calculate how many steps have been completed
            steps_completed = original_travel_time - resident.travel_time_remaining
            
            # Add progress within the current step (0 to 1)
            total_progress = steps_completed + progress
            
            # Calculate distance traveled so far (80 meters per step)
            distance_traveled = total_progress * 80.0
            
            # Clamp to total distance to avoid overshooting
            distance_traveled = min(distance_traveled, total_distance)
            
            # Find position along the path
            current_distance = 0
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                edge_data = self.graph.get_edge_data(node1, node2, 0)
                
                if 'length' in edge_data:
                    segment_length = edge_data['length']
                else:
                    x1, y1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y']
                    x2, y2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y']
                    segment_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                
                # Check if the target distance is within this segment
                if current_distance + segment_length >= distance_traveled:
                    # Interpolate within this segment
                    segment_progress = (distance_traveled - current_distance) / segment_length
                    segment_progress = max(0, min(1, segment_progress))  # Clamp to [0,1]
                    
                    # Get segment start and end positions
                    start_x = self.graph.nodes[node1]['x']
                    start_y = self.graph.nodes[node1]['y']
                    end_x = self.graph.nodes[node2]['x']
                    end_y = self.graph.nodes[node2]['y']
                    
                    # Interpolate position
                    x = start_x + segment_progress * (end_x - start_x)
                    y = start_y + segment_progress * (end_y - start_y)
                    
                    return x, y
                
                current_distance += segment_length
            
            # If we've gone through all segments, return the end position
            return self.graph.nodes[path[-1]]['x'], self.graph.nodes[path[-1]]['y']
            
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
                # TEMPORARY FEATURE: Check if resident is a tourist (casino-spawned)
                is_tourist = getattr(resident, 'is_tourist', False)
                
                if hasattr(resident, 'traveling') and resident.traveling:
                    # Get interpolated position for traveling agents
                    # Use progress to show position along the path within the current travel step
                    x, y = self._get_interpolated_position(resident, progress)
                    
                    # TEMPORARY: Use violet color for tourists, blue for regular residents
                    color = '#8A2BE2' if is_tourist else '#00BFFF'  # Violet for tourists, blue for regular
                    dot = self.ax.plot(x, y, 'o', color=color, markersize=4, markeredgecolor='black',
                                        markeredgewidth=0.5, alpha=0.8)[0]  
                else:
                    # For stationary agents, use the current node position
                    if hasattr(resident, 'current_node'):
                        x, y = self.graph.nodes[resident.current_node]['x'], self.graph.nodes[resident.current_node]['y']
                        
                        # TEMPORARY: Use violet color for tourists, blue for regular residents
                        color = '#8A2BE2' if is_tourist else '#00BFFF'  # Violet for tourists, blue for regular
                        dot = self.ax.plot(x, y, 'o', color=color, markersize=4,
                                           markeredgecolor='black', markeredgewidth=0.5, alpha=0.8)[0]
                    else:
                        # Use geometry if available
                        x, y = resident.geometry.x, resident.geometry.y
                        
                        # TEMPORARY: Use violet color for tourists, blue for regular residents
                        color = '#8A2BE2' if is_tourist else '#00BFFF'  # Violet for tourists, blue for regular
                        dot = self.ax.plot(x, y, 'o', color=color, markersize=3, alpha=0.7)[0]
                
                self.agent_dots.append(dot)
                
                # Register with interactive enhancer if available
                if self.interactive_enhancer:
                    self.interactive_enhancer.register_layer_artist(dot, 'agents')
                    
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
            "casino": 0,
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
                
                # Register with interactive enhancer if available
                if self.interactive_enhancer:
                    self.interactive_enhancer.register_layer_artist(marker, 'pois')
                    
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
            "casino": "Casinos",
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
        
        # TEMPORARY: Add tourists to legend
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#8A2BE2', markersize=8,
                      label='Tourists')
        )
        
        # Add environment elements to legend if they exist
        if self.residential_buildings is not None and not self.residential_buildings.empty:
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor='#d3d3d3', edgecolor='gray',
                             label='Residential Buildings')
            )
        
        if self.water_bodies is not None and not self.water_bodies.empty:
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor='#4FC3F7', edgecolor='#0277BD',
                             alpha=0.7, label='Water Bodies')
            )
        
        if self.forests is not None and not self.forests.empty:
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor='#66BB6A', edgecolor='#2E7D32',
                             alpha=0.6, label='Forests & Green Areas')
            )
        
        if self.cliffs is not None and not self.cliffs.empty:
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor='#8D6E63', edgecolor='#5D4037',
                             alpha=0.8, label='Cliffs & Barriers')
            )
        
        # Create the legend
        self.ax.legend(handles=legend_elements, 
                      loc='upper right',
                      title="Map Elements",
                      framealpha=0.8)
        
        # Adjust figure to make room for legend
        self.fig.tight_layout()
    
    def _add_scale_bar(self):
        """Add a scale bar to the bottom left of the map"""
        # Get the current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Calculate a reasonable scale bar length (about 10% of map width)
        map_width = xlim[1] - xlim[0]
        scale_length_deg = map_width * 0.1
        
        # Get the center latitude of the current map for accurate conversion
        center_lat = (ylim[0] + ylim[1]) / 2
        
        # Convert to meters using the actual latitude
        # At any latitude, 1 degree longitude ≈ 111,000 * cos(latitude) meters
        scale_length_m = scale_length_deg * 111000 * np.cos(np.radians(center_lat))
        
        # Round to a nice number
        if scale_length_m > 5000:
            scale_length_m = round(scale_length_m / 1000) * 1000  # Round to nearest km
            scale_text = f"{int(scale_length_m/1000)} km"
        elif scale_length_m > 1000:
            scale_length_m = round(scale_length_m / 500) * 500   # Round to nearest 500m
            scale_text = f"{int(scale_length_m)} m"
        else:
            scale_length_m = round(scale_length_m / 100) * 100   # Round to nearest 100m
            scale_text = f"{int(scale_length_m)} m"
        
        # Convert back to degrees for plotting using the same latitude
        scale_length_deg = scale_length_m / (111000 * np.cos(np.radians(center_lat)))
        
        # Position the scale bar (bottom left corner with margins)
        scale_x = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        scale_y = ylim[0] + (ylim[1] - ylim[0]) * 0.05
        
        # Draw the scale bar
        scale_bar = self.ax.plot([scale_x, scale_x + scale_length_deg], 
                                [scale_y, scale_y], 
                                'k-', linewidth=3)[0]
        
        # Add tick marks at the ends
        tick_height = (ylim[1] - ylim[0]) * 0.005
        left_tick = self.ax.plot([scale_x, scale_x], 
                                [scale_y - tick_height, scale_y + tick_height], 
                                'k-', linewidth=2)[0]
        right_tick = self.ax.plot([scale_x + scale_length_deg, scale_x + scale_length_deg], 
                                 [scale_y - tick_height, scale_y + tick_height], 
                                 'k-', linewidth=2)[0]
        
        # Add scale text
        text_y = scale_y + (ylim[1] - ylim[0]) * 0.01
        scale_label = self.ax.text(scale_x + scale_length_deg/2, text_y, scale_text,
                                  ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add white outline to text for better visibility
        scale_label.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
    
    def _add_north_arrow(self):
        """Add a north arrow above the scale bar in the bottom left area"""
        # Get the current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Position the north arrow above the scale bar (bottom left area)
        # Scale bar is at 5% from edges, so place arrow at same x but higher y
        arrow_x = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        arrow_y = ylim[0] + (ylim[1] - ylim[0]) * 0.12  # Above scale bar area
        
        # Calculate arrow size
        arrow_length = (ylim[1] - ylim[0]) * 0.04
        arrow_width = (xlim[1] - xlim[0]) * 0.01
        
        # Create north arrow using FancyArrowPatch
        arrow = FancyArrowPatch((arrow_x, arrow_y - arrow_length/2),
                               (arrow_x, arrow_y + arrow_length/2),
                               arrowstyle='-|>', 
                               mutation_scale=20,
                               color='black',
                               linewidth=2)
        self.ax.add_patch(arrow)
        
        # Add 'N' label
        n_label = self.ax.text(arrow_x, arrow_y + arrow_length/2 + (ylim[1] - ylim[0]) * 0.015, 
                              'N',
                              ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add white outline to text for better visibility
        n_label.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
        
        # Add a background circle for better visibility
        circle = plt.Circle((arrow_x, arrow_y), arrow_length * 0.7, 
                          fill=True, facecolor='white', edgecolor='black', 
                          alpha=0.8, linewidth=1)
        self.ax.add_patch(circle)