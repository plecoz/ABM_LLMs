import mesa
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
import random
from agents.resident import Resident
from agents.poi import POI

class FifteenMinuteCity(Model):
    def __init__(self, graph, pois, num_agents=10):
        super().__init__()  # Mesa 3.x model initialization
        
        self.graph = graph
        self.pois = pois
        self.schedule = RandomActivation(self)
        self.grid = NetworkGrid(graph)
        
        # Create agents
        for i in range(num_agents):
            home_node = random.choice(list(graph.nodes()))
            resident = Resident(
                unique_id=i,  # Can be integer in Mesa 3.x
                model=self,
                home_node=home_node
            )
            self.grid.place_agent(resident, home_node)
    def step(self):
        """Advance the model by one step"""
        self.schedule.step()  # This calls step() on all agents

    def register_agent(self, agent):
        """Explicit registration (optional but recommended)"""
        self.schedule.add(agent)