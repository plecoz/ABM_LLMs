class AgentPlotter:
    def __init__(self, base_map):
        self.base = base_map
        self.resident_markers = []
        self.poi_markers = []
        self.organizations_markers = []
    
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
    def plot_organizations(self, organizations, graph):
        """Plot residents as blue circles."""
        for organization in organizations:
            x, y = graph.nodes[organization.current_node]['x'], graph.nodes[organization.current_node]['y']
            m = self.base.ax.plot(x, y, 'o', color='green', markersize=8)[0]
            self.organizations_markers.append(m)