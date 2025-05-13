import mesa
from mesa import Model
from mesa.space import NetworkGrid
import random
from agents.resident import Resident
from agents.organizationagent import OrganizationAgent
from agents.poi import POI
import networkx as nx
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
import json
import logging
import uuid
import numpy as np
import pandas as pd
from mesa_geo import GeoSpace
from mesa.datacollection import DataCollector


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Custom scheduler implementation to replace RandomActivation
class CustomRandomActivation:
    def __init__(self, model):
        self.model = model
        self.agents = []
        self.steps = 0
    
    def add(self, agent):
        """Add an agent to the scheduler"""
        self.agents.append(agent)
    
    def remove(self, agent):
        """Remove an agent from the scheduler"""
        self.agents.remove(agent)
    
    def step(self):
        """Execute the step of all agents, one at a time, in random order"""
        random.shuffle(self.agents)
        for agent in self.agents:
            agent.step()
        self.steps += 1

class GeometryEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that can handle Shapely geometry objects.
    """
    def default(self, obj):
        if isinstance(obj, BaseGeometry):
            # Convert Shapely geometry to a dictionary representation
            if hasattr(obj, 'x') and hasattr(obj, 'y'):  # Point objects
                return {'type': 'Point', 'coordinates': [obj.x, obj.y]}
            elif hasattr(obj, '__geo_interface__'):  # Use the __geo_interface__ if available
                return obj.__geo_interface__
            else:  # Fallback for other geometry types
                try:
                    return {'type': obj.geom_type, 'wkt': obj.wkt}
                except:
                    return str(obj)
        # Handle tuples with geometry objects (like in location_history)
        elif isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], BaseGeometry):
            # This handles (step, geometry) tuples in location_history
            return [obj[0], self.default(obj[1])]
        return super().default(obj)

class FifteenMinuteCity(Model):
    def __init__(self, graph, pois, num_residents, num_organizations, **kwargs):
        super().__init__()  # Mesa 3.x model initialization
        
        # Initialize logger
        self.logger = logging.getLogger("FifteenMinuteCity")
        
        self.graph = graph
        self.pois = pois
        self.grid = NetworkGrid(graph)
        
        # Add a step counter
        self.step_count = 0
        
        # Initialize lists to track agents
        self.residents = []
        self.organizations = []
        self.all_agents = []
        self.communications = []  # Store communications

        # Initialize demographics with default values
        self.demographics = {
            "age_distribution": {"0-18": 0.2, "19-35": 0.3, "36-65": 0.4, "65+": 0.1},
            "gender_distribution": {"male": 0.49, "female": 0.49, "other": 0.02},
            "income_distribution": {"low": 0.3, "medium": 0.5, "high": 0.2},
            "education_distribution": {
                "no_education": 0.1,
                "primary": 0.2,
                "high_school": 0.4,
                "bachelor": 0.2,
                "master": 0.08,
                "phd": 0.02
            }
        }
        
        # Load demographics if path is provided
        demographics_path = kwargs.get('demographics_path')
        if demographics_path:
            self._load_demographics(demographics_path)

        self.random = random.Random(kwargs.get('seed', None))
        
        # Set up scheduler and spatial environment
        self.schedule = CustomRandomActivation(self)
        self.space = GeoSpace()
        
        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Total Agents": lambda m: m.get_agent_count(),
                "Person Agents": lambda m: m.get_agent_count(agent_type=Resident),
                "Organization Agents": lambda m: m.get_agent_count(agent_type=OrganizationAgent)
            },
            agent_reporters={
                "Position": lambda a: (a.geometry.x, a.geometry.y),
                "Type": lambda a: a.__class__.__name__
            }
        )
        
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

            agent_props = self._generate_agent_properties()

            resident = Resident(
                model=self,
                unique_id=i,
                geometry=point_geometry,
                home_node=home_node,
                accessible_nodes=accessible_nodes,
                **agent_props
            )
            self.grid.place_agent(resident, home_node)
            self.schedule.add(resident)  # Add to our custom scheduler
            self.residents.append(resident)
            self.all_agents.append(resident)

            org_types = kwargs.get('org_types', ['business', 'government', 'school'])
        
        for i in range(num_organizations):
            home_node = random.choice(list(graph.nodes()))
            # Get coordinates from the node
            node_coords = self.graph.nodes[home_node]

            # Create a Point geometry from the coordinates
            point_geometry = Point(node_coords['x'], node_coords['y'])
            
            org_type = self.random.choice(org_types)

            organization = OrganizationAgent(
                model=self,
                unique_id=i + num_residents,  # Ensure unique IDs
                geometry=point_geometry,
                current_node=home_node,
                org_type=org_type
            )
            self.grid.place_agent(organization, home_node)
            self.schedule.add(organization)  # Add to our custom scheduler
            self.organizations.append(organization)
            self.all_agents.append(organization)

        self._initialize_social_networks(kwargs.get('social_network_density', 0.1))
        self.logger.info(f"Generated {num_residents} resident agents and {num_organizations} organization agents")

    # Add a method to get agent by ID
    def get_agent_by_id(self, agent_id):
        """Get an agent by its ID"""
        for agent in self.all_agents:
            if agent.unique_id == agent_id:
                return agent
        return None

    def get_agent_count(self, agent_type=None):
        """Get the count of agents matching a specific type"""
        if agent_type is None:
            return len(self.all_agents)
        return sum(1 for agent in self.all_agents if isinstance(agent, agent_type))

    def step(self):
        """Advance the model by one step"""
        # Increment step counter
        self.step_count += 1
        
        # Use the scheduler to step all agents
        self.schedule.step()
        
        self.datacollector.collect(self)
        
        # Process any global model dynamics
        self._process_model_dynamics()

    def _process_model_dynamics(self):
        """
        Process any global model dynamics.
        This can include environmental changes, global events, etc.
        """
        # This is a placeholder. You can add model-wide dynamics here.
        pass
    
    def _load_demographics(self, demographics_path):
        """
        Load demographic data from a JSON file.
        
        Args:
            demographics_path: Path to the JSON file with demographic data
        """
        try:
            with open(demographics_path, 'r') as f:
                self.demographics = json.load(f)
            self.logger.info("Loaded demographics data")
        except Exception as e:
            self.logger.error(f"Error loading demographics: {e}")
            # Set default demographics
            self.demographics = {
                "age_distribution": {"0-18": 0.2, "19-35": 0.3, "36-65": 0.4, "65+": 0.1},
                "gender_distribution": {"male": 0.49, "female": 0.49, "other": 0.02},
                "income_distribution": {"low": 0.3, "medium": 0.5, "high": 0.2},
                "education_distribution": {
                    "no_education": 0.1,
                    "primary": 0.2,
                    "high_school": 0.4,
                    "bachelor": 0.2,
                    "master": 0.08,
                    "phd": 0.02
                }
            }
    

    def _generate_agent_properties(self):
        """
        Generate agent properties based on demographic distributions.
        
        Returns:
            Dictionary of agent properties
        """
        if not self.demographics:
            return {}
        
        props = {}
        
        # Generate age
        age_dist = self.demographics.get('age_distribution', {})
        age_group = self._sample_from_distribution(age_dist)
        
        # Convert age group to actual age
        if age_group == "0-18":
            props['age'] = self.random.randint(0, 18)
        elif age_group == "19-35":
            props['age'] = self.random.randint(19, 35)
        elif age_group == "36-65":
            props['age'] = self.random.randint(36, 65)
        else:  # 65+
            props['age'] = self.random.randint(65, 90)
        
        # Generate gender
        gender_dist = self.demographics.get('gender_distribution', {})
        props['gender'] = self._sample_from_distribution(gender_dist)
        
        # Generate income
        income_dist = self.demographics.get('income_distribution', {})
        income_level = self._sample_from_distribution(income_dist)
        
        # Convert income level to actual income
        if income_level == "low":
            props['income'] = self.random.randint(10000, 30000)
        elif income_level == "medium":
            props['income'] = self.random.randint(30001, 100000)
        else:  # high
            props['income'] = self.random.randint(100001, 500000)
        
        # Generate education
        education_dist = self.demographics.get('education_distribution', {})
        props['education'] = self._sample_from_distribution(education_dist)
        
        return props
    

    def _sample_from_distribution(self, distribution):
        """
        Sample a value from a probability distribution.
        
        Args:
            distribution: Dictionary mapping values to probabilities
            
        Returns:
            A sampled value from the distribution
        """
        if not distribution:
            return None
            
        items = list(distribution.keys())
        probabilities = list(distribution.values())
        
        # Normalize probabilities if they don't sum to 1
        prob_sum = sum(probabilities)
        if prob_sum != 1.0 and prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]
        
        return self.random.choices(items, probabilities)[0]
    

    def _initialize_social_networks(self, density=0.1):
        """
        Initialize social networks between resident agents.
        
        Args:
            density: Probability of connection between any two agents
        """
        # For each agent, connect to others with probability 'density'
        for agent in self.residents:
            for other in self.residents:
                # Don't connect to self and only process each pair once
                if agent.unique_id != other.unique_id and self.random.random() < density:
                    agent.add_to_social_network(other.unique_id)
                    # Bidirectional connection
                    other.add_to_social_network(agent.unique_id)

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
        
        # Check all agents
        for other in self.all_agents:
            if other.unique_id != agent.unique_id:
                if agent.geometry.distance(other.geometry) <= distance:
                    nearby_agents.append(other)
        
        return nearby_agents

    def record_communication(self, message):
        """
        Record a communication between agents.
        
        Args:
            message: The message object to record
        """
        # In a full implementation, this would store the message in a database or log
        # For now, we'll just store it in the communications list
        self.communications.append(message)

    """
    def register_agent(self, agent):
        #Explicit registration (optional but recommended)
        self.schedule.add(agent)
    """