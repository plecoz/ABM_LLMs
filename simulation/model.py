import mesa
from mesa import Model
from mesa.space import NetworkGrid
import random
from agents.resident import Resident
from agents.organizationagent import OrganizationAgent
from agents.poi import POI
import networkx as nx
from shapely.geometry import Point

class FifteenMinuteCity(Model):
    def __init__(self, graph, pois, num_residents, num_organizations):
        super().__init__()  # Mesa 3.x model initialization
        
        self.graph = graph
        self.pois = pois
        #self.schedule = RandomActivation(self)
        #self.agents.shuffle_do("step")
        self.grid = NetworkGrid(graph)
        
        # Add a step counter
        self.step_count = 0
        
        # Initialize lists to track agents
        self.residents = []
        self.organizations = []
        self.all_agents = []
        
        # Create agents
        for i in range(num_residents):
            home_node = random.choice(list(graph.nodes()))
            # Get coordinates from the node
            node_coords = self.graph.nodes[home_node]
            # Create a Point geometry from the coordinates
            point_geometry = Point(node_coords['x'], node_coords['y'])
            
            # Calculate all nodes within 1km
            accessible_nodes = dict(nx.single_source_dijkstra_path_length(
                graph, home_node, cutoff=1000, weight='length'
            ))
            resident = Resident(
                model=self,
                unique_id=i,
                geometry=point_geometry,
                home_node=home_node,
                accessible_nodes=accessible_nodes
            )
            self.grid.place_agent(resident, home_node)
            self.residents.append(resident)
            self.all_agents.append(resident)
        
        for i in range(num_organizations):
            home_node = random.choice(list(graph.nodes()))
            # Get coordinates from the node
            node_coords = self.graph.nodes[home_node]
            # Create a Point geometry from the coordinates
            point_geometry = Point(node_coords['x'], node_coords['y'])
            
            # Calculate all nodes within 1km

            organization = OrganizationAgent(
                model=self,
                unique_id=i,
                geometry=point_geometry,
                current_node=home_node  # Pass the home_node as current_node
            )
            self.grid.place_agent(organization, home_node)
            self.organizations.append(organization)
            self.all_agents.append(organization)


    def step(self):
        """Advance the model by one step"""
        # Increment step counter
        self.step_count += 1
        
        # Step all residents
        for resident in self.residents:
            resident.step()
        
        # Step all organizations
        for organization in self.organizations:
            organization.step()

    def get_nearby_agents(self, agent, distance=1.0):
        """
        Get agents within a certain distance of the given agent.
        
        Args:
            agent: The agent to find nearby agents for
            distance: The maximum distance to search
            
        Returns:
            List of agents within the specified distance
        """
        nearby_agents = []
        
        # Check all residents and organizations
        
        for other in self.all_agents:
            if other.unique_id != agent.unique_id:
                if agent.geometry.distance(other.geometry) <= distance:
                    nearby_agents.append(other)
        
        return nearby_agents
"""
    def register_agent(self, agent):
        #Explicit registration (optional but recommended)
        self.schedule.add(agent)
"""
"""
    def record_communication(self, message):
        
        Record a communication between agents.
        
        Args:
            message: The message object to record
        
        # In a full implementation, this would store the message in a database or log
        # For now, we'll just print it
        if hasattr(self, 'communications'):
            self.communications.append(message)
        else:
            self.communications = [message]
"""