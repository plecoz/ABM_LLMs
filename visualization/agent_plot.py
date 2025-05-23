class AgentPlotter:
    def __init__(self, base_map):
        self.base = base_map
        self.resident_markers = []
        self.poi_markers = []
    
    def plot_pois(self, pois, graph):
        """Plot POIs as red squares."""
        for poi in pois:
            x, y = graph.nodes[poi.node_id]['x'], graph.nodes[poi.node_id]['y']
            m = self.base.ax.plot(x, y, 's', color='red', markersize=8)[0]
            self.poi_markers.append(m)
    
    def plot_residents(self, residents, graph):
        """Plot residents as blue circles."""
        for resident in residents:
            x, y = graph.nodes[resident.current_node]['x'], graph.nodes[resident.current_node]['y']
            m = self.base.ax.plot(x, y, 'o', color='deepskyblue', markersize=6)[0]
            self.resident_markers.append(m)
            
    def plot_poi_agents(self, poi_agents):
        """Plot POI agents with colors and shapes based on their categories."""
        # Define colors and shapes for different POI categories
        category_styles = {
            'daily_living': {'color': '#FF9800', 'marker': 's'},  # Orange square
            'healthcare': {'color': '#F44336', 'marker': 'h'},    # Red hexagon
            'education': {'color': '#2196F3', 'marker': '^'},     # Blue triangle
            'entertainment': {'color': '#4CAF50', 'marker': '*'}, # Green star
            'transportation': {'color': '#000000', 'marker': 'd'}, # Black diamond
            'other': {'color': '#9E9E9E', 'marker': 'o'}          # Gray circle
        }
        
        for poi in poi_agents:
            category = poi.category if hasattr(poi, 'category') else 'other'
            style = category_styles.get(category, category_styles['other'])
            
            x, y = poi.geometry.x, poi.geometry.y
            
            # Special case for transportation (bus stops)
            if category == 'transportation':
                size = 4  # Smaller size for bus stops
            else:
                size = 6  # Standard size for other POIs
                
            m = self.base.ax.plot(x, y, style['marker'], color=style['color'], 
                                 markersize=size, alpha=0.8)[0]
            self.poi_markers.append(m)