import random
import mesa
from mesa.agent import Agent
import networkx as nx

class Resident(mesa.Agent):
    """An agent representing a resident in the 15-minute city."""
    def __init__(self, unique_id, model, home_node):
        super().__init__(unique_id, model)
        self.home_node = home_node
        self.current_node = home_node
        self.visited_pois = []
        self.mobility_mode = "walk"  # or "bike"

    def move_to_poi(self, poi_type):
        """Move to a POI if within 15-minute range (1km walk)."""
        possible_pois = self.model.pois[poi_type]
        if not possible_pois:
            return False
        
        target_poi = random.choice(possible_pois)
        try:
            path = self.model.graph.shortest_path(self.current_node, target_poi)
            distance = self.model.graph.shortest_path_length(self.home_node, target_poi, weight="length")
            if distance <= 1000:  # 1km ~ 15 min walk
                self.current_node = target_poi
                self.visited_pois.append(target_poi)
                return True
        except nx.NetworkXNoPath:
            pass
        return False
    

    def step(self):
        """Move to a random POI type (school/shop/park)"""
        poi_type = random.choice(["school", "shop", "park"])
        if poi_type in self.model.pois:
            self.move_to_poi(poi_type)
