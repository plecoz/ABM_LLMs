import mesa

class POI(mesa.Agent):
    """A Point of Interest (shop, school, etc.)."""
    def __init__(self, unique_id, model, node_id, poi_type):
        super().__init__(unique_id, model)
        self.node_id = node_id
        self.poi_type = poi_type