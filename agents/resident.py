from mesa.agent import Agent
import random
import networkx as nx

class Resident(Agent):
    def __init__(self, unique_id, model, home_node, accessible_nodes):
        """
        Proper initialization for Mesa 3.x:
        - model MUST be the first argument to parent class
        - unique_id is set separately
        """
        # Parent class gets model only
        super().__init__(model)
        
        # Then we set our unique_id
        self.unique_id = unique_id
        
        # Custom attributes
        self.home_node = home_node
        self.current_node = home_node
        self.accessible_nodes = accessible_nodes
        self.visited_pois = []
        self.mobility_mode = "walk"
        
        # Register with model (automatic in Mesa, but explicit here)
        #model.register_agent(self)

    def move_to_poi(self, poi_type):
        """Improved movement with distance check"""
        if not self.model.pois.get(poi_type):
            return False
        
        valid_pois = [n for n in self.model.pois[poi_type] 
                     if n in self.accessible_nodes]
        
        if not valid_pois:
            return False
            
        try:
           #target = random.choice(self.model.pois[poi_type])
            target = random.choice(valid_pois)
            """
            distance = nx.shortest_path_length(
                self.model.graph,
                self.current_node,
                target,
                weight="length"
            )
            if distance <= 1000:  # 15-min walk threshold (~1km)
                self.current_node = target
                self.visited_pois.append(target)
                return True
            """
            self.current_node = target
            self.visited_pois.append(target)
        except (nx.NetworkXNoPath, KeyError):
            pass
        return False

    def step(self):
        if self.model.pois:
            self.move_to_poi(random.choice(list(self.model.pois.keys())))
