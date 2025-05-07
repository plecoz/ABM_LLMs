import mesa
import random
from agents.resident import Resident
from agents.poi import POI

class FifteenMinuteCity(mesa.Model):
    """Mesa model for the 15-minute city simulation."""
    def __init__(self, graph, pois, num_agents=10):  # Add num_agents parameter
        # ... (existing code)
        for i in range(num_agents):  # Use parameter instead of hardcoded 100
            home_node = random.choice(list(graph.nodes()))
            resident = Resident(i, self, home_node)
            self.schedule.add(resident)
        
        # Create residents
        for i in range(100):
            home_node = random.choice(list(graph.nodes()))
            resident = Resident(i, self, home_node)
            self.schedule.add(resident)
        
        # Create POIs (optional)
        poi_id = 1000
        for poi_type, nodes in pois.items():
            for node in nodes:
                poi = POI(poi_id, self, node, poi_type)
                poi_id += 1
                self.schedule.add(poi)

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()