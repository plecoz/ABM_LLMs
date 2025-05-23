import networkx as nx

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
        """Plot residents as blue circles with interpolated positions for traveling agents."""
        for resident in residents:
            try:
                if hasattr(resident, 'traveling') and resident.traveling:
                    # Get interpolated position for traveling agents
                    x, y = self._get_interpolated_position(resident, graph)
                    m = self.base.ax.plot(x, y, 'o', color='#FF5722', markersize=6, alpha=0.8)[0]  # Orange for traveling
                else:
                    # For stationary agents, use the current node position
                    if hasattr(resident, 'current_node'):
                        x, y = graph.nodes[resident.current_node]['x'], graph.nodes[resident.current_node]['y']
                        m = self.base.ax.plot(x, y, 'o', color='deepskyblue', markersize=6, alpha=0.7)[0]
                    else:
                        # Use geometry if available
                        x, y = resident.geometry.x, resident.geometry.y
                        m = self.base.ax.plot(x, y, 'o', color='deepskyblue', markersize=6, alpha=0.7)[0]
                
                self.resident_markers.append(m)
            except Exception as e:
                print(f"Error plotting resident {resident.unique_id}: {e}")
    
    def _get_interpolated_position(self, resident, graph):
        """
        Calculate the interpolated position of a traveling resident along the path.
        
        Args:
            resident: The resident agent that is traveling
            graph: The network graph
            
        Returns:
            Tuple (x, y) of the current interpolated position
        """
        # If not traveling or missing necessary attributes, return current position
        if not resident.traveling or not hasattr(resident, 'current_node') or not hasattr(resident, 'destination_node'):
            x = resident.geometry.x
            y = resident.geometry.y
            return x, y
            
        try:
            # Get start and end nodes
            start_node = resident.current_node
            end_node = resident.destination_node
            
            # Calculate the shortest path between nodes
            path = nx.shortest_path(graph, start_node, end_node, weight='length')
            
            # If path has only one node (start == end), return that position
            if len(path) <= 1:
                return graph.nodes[start_node]['x'], graph.nodes[start_node]['y']
                
            # Calculate total travel time and remaining time
            total_time = resident.travel_time_remaining + 1  # +1 because we've already decremented by 1
            progress = 1 - (resident.travel_time_remaining / total_time)  # 0 to 1 progress
            
            # If we're at the beginning or end of the journey
            if progress <= 0:
                return graph.nodes[start_node]['x'], graph.nodes[start_node]['y']
            elif progress >= 1:
                return graph.nodes[end_node]['x'], graph.nodes[end_node]['y']
                
            # Calculate cumulative distances along the path
            distances = [0]  # Start with 0 distance
            total_distance = 0
            
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                # Get edge data
                edge_data = graph.get_edge_data(node1, node2, 0)  # 0 is the default key for MultiDiGraph
                # Get length or calculate Euclidean distance
                if 'length' in edge_data:
                    distance = edge_data['length']
                else:
                    x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
                    x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                
                total_distance += distance
                distances.append(total_distance)
            
            # Normalize distances to 0-1 range
            if total_distance > 0:
                distances = [d / total_distance for d in distances]
            
            # Find which segment the agent is on
            target_distance = progress
            segment_idx = 0
            while segment_idx < len(distances) - 1 and distances[segment_idx + 1] < target_distance:
                segment_idx += 1
                
            # If we're at the last node
            if segment_idx >= len(distances) - 1:
                return graph.nodes[path[-1]]['x'], graph.nodes[path[-1]]['y']
                
            # Calculate interpolation within the segment
            start_dist = distances[segment_idx]
            end_dist = distances[segment_idx + 1]
            
            if end_dist == start_dist:  # Avoid division by zero
                segment_progress = 0
            else:
                segment_progress = (target_distance - start_dist) / (end_dist - start_dist)
                
            # Get segment start and end positions
            start_x = graph.nodes[path[segment_idx]]['x']
            start_y = graph.nodes[path[segment_idx]]['y']
            end_x = graph.nodes[path[segment_idx + 1]]['x']
            end_y = graph.nodes[path[segment_idx + 1]]['y']
            
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
                return graph.nodes[resident.current_node]['x'], graph.nodes[resident.current_node]['y']
            
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