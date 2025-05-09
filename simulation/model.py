import mesa
from mesa import Model
#from mesa.time import RandomActivation
from mesa.space import NetworkGrid
import random
from agents.resident import Resident
from agents.poi import POI
import networkx as nx

class FifteenMinuteCity(Model):
    def __init__(self, graph, pois, num_agents=10):
        super().__init__()  # Mesa 3.x model initialization
        
        self.graph = graph
        self.pois = pois
        #self.schedule = RandomActivation(self)
        #self.agents.shuffle_do("step")
        self.grid = NetworkGrid(graph)
        
        # Create agents
        for i in range(num_agents):
            home_node = random.choice(list(graph.nodes()))
                        # Calculate all nodes within 1km
            accessible_nodes = dict(nx.single_source_dijkstra_path_length(
                graph, home_node, cutoff=1000, weight='length'
            ))
            resident = Resident(
                unique_id=i,  # Can be integer in Mesa 3.x
                model=self,
                home_node=home_node,
                accessible_nodes=accessible_nodes
            )
            self.grid.place_agent(resident, home_node)


    def step(self):
        """Advance the model by one step"""
        #self.schedule.step()
        self.agents.shuffle_do("step")  # This calls step() on all agents
"""
    def register_agent(self, agent):
        #Explicit registration (optional but recommended)
        self.schedule.add(agent)
"""
