"""
Interactive matplotlib enhancements for the Macau ABM visualization.
Adds zoom, pan, and better building visualization while keeping existing functionality.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import CheckButtons
import numpy as np
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from shapely.geometry import Point

class InteractiveEnhancer:
    """
    Enhances matplotlib plots with interactive features like zoom, pan, and layer toggles.
    """
    
    def __init__(self, ax, fig=None):
        self.ax = ax
        self.fig = fig or ax.figure
        self.original_xlim = None
        self.original_ylim = None
        
        # Pan state
        self.pan_active = False
        self.pan_start = None
        
        # Layer visibility
        self.layers = {
            'buildings': True,
            'agents': True,
            'pois': True,
            'streets': True
        }
        
        # Store layer artists for toggling
        self.layer_artists = {
            'buildings': [],
            'agents': [],
            'pois': [],
            'streets': []
        }
        
        # Setup interactive features
        self._setup_interaction()
        
    def _setup_interaction(self):
        """Setup mouse and keyboard interactions."""
        # Store original limits
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Add toolbar if not present
        if hasattr(self.fig.canvas, 'toolbar'):
            self.fig.canvas.toolbar_visible = True
            
    def _on_scroll(self, event):
        """Handle mouse wheel zoom."""
        if event.inaxes != self.ax:
            return
            
        # Get current limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        # Get mouse position
        xdata = event.xdata
        ydata = event.ydata
        
        if xdata is None or ydata is None:
            return
            
        # Zoom factor
        zoom_factor = 0.8 if event.button == 'up' else 1.25
        
        # Calculate new limits
        x_range = (cur_xlim[1] - cur_xlim[0]) * zoom_factor
        y_range = (cur_ylim[1] - cur_ylim[0]) * zoom_factor
        
        new_xlim = [xdata - x_range/2, xdata + x_range/2]
        new_ylim = [ydata - y_range/2, ydata + y_range/2]
        
        # Set new limits
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        # Redraw
        self.fig.canvas.draw_idle()
        
    def _on_click(self, event):
        """Handle mouse click for panning."""
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:  # Left click
            self.pan_active = True
            self.pan_start = (event.xdata, event.ydata)
            
    def _on_release(self, event):
        """Handle mouse release."""
        if event.button == 1:  # Left click
            self.pan_active = False
            self.pan_start = None
            
    def _on_motion(self, event):
        """Handle mouse motion for panning."""
        if not self.pan_active or self.pan_start is None:
            return
            
        if event.inaxes != self.ax:
            return
            
        # Calculate pan distance
        dx = event.xdata - self.pan_start[0]
        dy = event.ydata - self.pan_start[1]
        
        # Get current limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        # Apply pan
        new_xlim = [cur_xlim[0] - dx, cur_xlim[1] - dx]
        new_ylim = [cur_ylim[0] - dy, cur_ylim[1] - dy]
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        # Redraw
        self.fig.canvas.draw_idle()
        
    def _on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'r':  # Reset view
            self.reset_view()
        elif event.key == 'h':  # Show help
            self.show_help()
            
    def reset_view(self):
        """Reset view to original limits."""
        if self.original_xlim and self.original_ylim:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.fig.canvas.draw_idle()
            
    def show_help(self):
        """Show help dialog with keyboard shortcuts."""
        help_text = """
        Interactive Controls:
        
        Mouse:
        • Scroll wheel: Zoom in/out
        • Left click + drag: Pan
        
        Keyboard:
        • R: Reset view
        • H: Show this help
        
        Toolbar:
        • Use matplotlib toolbar for additional tools
        """
        
        # Create help dialog
        fig_help = plt.figure(figsize=(6, 4))
        fig_help.suptitle("Interactive Controls Help")
        ax_help = fig_help.add_subplot(111)
        ax_help.text(0.05, 0.95, help_text, transform=ax_help.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax_help.set_xlim(0, 1)
        ax_help.set_ylim(0, 1)
        ax_help.axis('off')
        plt.tight_layout()
        plt.show()
        
    def add_layer_controls(self):
        """Add layer visibility controls."""
        # Create checkboxes for layer control
        if hasattr(self.fig, 'layer_controls'):
            return  # Already added
            
        # Add space for controls
        self.fig.subplots_adjust(left=0.15)
        
        # Create checkbox axes
        checkbox_ax = self.fig.add_axes([0.02, 0.7, 0.1, 0.2])
        
        # Create checkboxes
        labels = ['Buildings', 'Agents', 'POIs', 'Streets']
        visibility = [self.layers[key] for key in ['buildings', 'agents', 'pois', 'streets']]
        
        self.checkbox = CheckButtons(checkbox_ax, labels, visibility)
        self.checkbox.on_clicked(self._toggle_layer)
        
        # Mark as added
        self.fig.layer_controls = True
        
    def _toggle_layer(self, label):
        """Toggle layer visibility."""
        layer_map = {
            'Buildings': 'buildings',
            'Agents': 'agents', 
            'POIs': 'pois',
            'Streets': 'streets'
        }
        
        layer_key = layer_map.get(label)
        if layer_key:
            self.layers[layer_key] = not self.layers[layer_key]
            self._update_layer_visibility(layer_key)
            
    def _update_layer_visibility(self, layer_key):
        """Update visibility of layer artists."""
        visible = self.layers[layer_key]
        
        for artist in self.layer_artists.get(layer_key, []):
            artist.set_visible(visible)
            
        self.fig.canvas.draw_idle()
        
    def register_layer_artist(self, artist, layer_type):
        """Register an artist with a layer for visibility control."""
        if layer_type in self.layer_artists:
            self.layer_artists[layer_type].append(artist)

class Enhanced3DBuildings:
    """
    Enhanced building visualization with height-based styling.
    """
    
    def __init__(self, ax):
        self.ax = ax
        self.building_patches = []
        
    def plot_3d_buildings(self, buildings_gdf, interactive_enhancer=None):
        """
        Plot buildings with height-based visualization.
        
        Args:
            buildings_gdf: GeoDataFrame with building data and processed_height column
            interactive_enhancer: InteractiveEnhancer instance for layer control
        """
        if buildings_gdf is None or len(buildings_gdf) == 0:
            print("No buildings data to plot")
            return
            
        print(f"Plotting {len(buildings_gdf)} buildings with height information...")
        
        # Create color map based on height
        heights = buildings_gdf['processed_height'].values
        
        # Create colormap
        norm = mcolors.Normalize(vmin=heights.min(), vmax=heights.max())
        cmap = plt.cm.viridis
        
        # Plot buildings
        patches_list = []
        colors = []
        
        for idx, building in buildings_gdf.iterrows():
            geom = building.geometry
            height = building['processed_height']
            
            try:
                # Convert geometry to patches
                if geom.geom_type == 'Polygon':
                    # Get exterior coordinates
                    x, y = geom.exterior.coords.xy
                    
                    # Check if we have enough coordinates
                    if len(x) >= 3 and len(y) >= 3:
                        # Create polygon patch
                        coords = list(zip(x, y))
                        if len(coords) >= 3:  # Need at least 3 points for a polygon
                            polygon = patches.Polygon(coords, closed=True)
                            patches_list.append(polygon)
                            colors.append(height)
                    
                elif geom.geom_type == 'MultiPolygon':
                    # Handle multipolygon
                    for poly in geom.geoms:
                        x, y = poly.exterior.coords.xy
                        
                        # Check if we have enough coordinates
                        if len(x) >= 3 and len(y) >= 3:
                            coords = list(zip(x, y))
                            if len(coords) >= 3:  # Need at least 3 points for a polygon
                                polygon = patches.Polygon(coords, closed=True)
                                patches_list.append(polygon)
                                colors.append(height)
                                
            except Exception as e:
                # Skip problematic geometries
                print(f"Warning: Skipping building {idx} due to geometry error: {e}")
                continue
        
        # Create patch collection
        if patches_list:
            collection = PatchCollection(patches_list, cmap=cmap, norm=norm, alpha=0.7)
            collection.set_array(np.array(colors))
            
            # Add to plot
            building_artist = self.ax.add_collection(collection)
            
            # Add colorbar
            cbar = plt.colorbar(collection, ax=self.ax, shrink=0.8)
            cbar.set_label('Building Height (m)', rotation=270, labelpad=20)
            
            # Register with interactive enhancer
            if interactive_enhancer:
                interactive_enhancer.register_layer_artist(building_artist, 'buildings')
                interactive_enhancer.register_layer_artist(cbar, 'buildings')
            
            self.building_patches.append(building_artist)
            
            print(f"Plotted {len(patches_list)} building polygons")
        else:
            print("No valid building polygons found to plot")
            
    def add_building_info_on_click(self, buildings_gdf):
        """Add click functionality to show building information."""
        def on_click(event):
            if event.inaxes != self.ax:
                return
                
            # Find clicked building
            x, y = event.xdata, event.ydata
            
            # Simple point-in-polygon check
            for idx, building in buildings_gdf.iterrows():
                if building.geometry.contains(Point(x, y)):
                    height = building['processed_height']
                    category = building.get('height_category', 'Unknown')
                    
                    # Show info
                    info_text = f"Building Height: {height:.1f}m\nCategory: {category}"
                    self.ax.text(x, y, info_text, bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="white", alpha=0.8),
                                fontsize=8, ha='center')
                    
                    # Redraw
                    self.ax.figure.canvas.draw_idle()
                    break
                    
        # Connect click event
        self.ax.figure.canvas.mpl_connect('button_press_event', on_click)

def enhance_existing_plot(ax, fig=None):
    """
    Enhance an existing matplotlib plot with interactive features.
    
    Args:
        ax: Matplotlib axes object
        fig: Matplotlib figure object (optional)
        
    Returns:
        InteractiveEnhancer instance
    """
    enhancer = InteractiveEnhancer(ax, fig)
    return enhancer

def create_enhanced_visualization(graph, buildings_gdf=None, title="Enhanced Macau Visualization"):
    """
    Create a complete enhanced visualization with all features.
    
    Args:
        graph: NetworkX graph of the street network
        buildings_gdf: GeoDataFrame with building data (optional)
        title: Plot title
        
    Returns:
        Tuple of (fig, ax, enhancer, building_plotter)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot street network (basic)
    import osmnx as ox
    ox.plot_graph(graph, ax=ax, show=False, close=False, 
                  node_size=0, edge_color='gray', edge_linewidth=0.5)
    
    # Add interactive enhancements
    enhancer = InteractiveEnhancer(ax, fig)
    
    # Add building visualization if provided
    building_plotter = None
    if buildings_gdf is not None:
        building_plotter = Enhanced3DBuildings(ax)
        building_plotter.plot_3d_buildings(buildings_gdf, enhancer)
        building_plotter.add_building_info_on_click(buildings_gdf)
    
    # Add layer controls
    enhancer.add_layer_controls()
    
    # Set title
    ax.set_title(title, fontsize=16, pad=20)
    
    # Add instructions
    instructions = "Use mouse wheel to zoom, drag to pan, press 'R' to reset, 'H' for help"
    fig.text(0.5, 0.02, instructions, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    return fig, ax, enhancer, building_plotter 