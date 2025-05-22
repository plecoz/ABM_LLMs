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
        """Plot POI agents as colored markers based on their category."""
        # Define colors for different POI categories
        category_colors = {
            'healthcare': 'red',
            'education': 'blue',
            'shopping': 'yellow',
            'recreation': 'green',
            'services': 'purple',
            'food': 'orange',
            'other': 'gray'
        }
        
        for poi in poi_agents:
            category = poi.category if hasattr(poi, 'category') else 'other'
            color = category_colors.get(category, 'gray')
            x, y = poi.geometry.x, poi.geometry.y
            m = self.base.ax.plot(x, y, 's', color=color, markersize=8)[0]
            self.poi_markers.append(m)