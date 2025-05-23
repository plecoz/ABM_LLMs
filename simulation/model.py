import mesa
from mesa import Model
from mesa.space import NetworkGrid
import random
from agents.resident import Resident
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
import geopandas as gpd


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
    def __init__(self, graph, pois, num_residents, **kwargs):
        super().__init__()  # Mesa 3.x model initialization
        
        # Initialize logger
        self.logger = logging.getLogger("FifteenMinuteCity")
        
        self.graph = graph
        self.pois = pois
        self.grid = NetworkGrid(graph)
        
        # Add a step counter
        self.step_count = 0
        
        # Add time simulation
        self.hour_of_day = kwargs.get('start_hour', 8)  # Start at 8 AM by default
        self.day_of_week = kwargs.get('start_day', 0)  # Start on Monday (0) by default
        self.day_count = 0
        
        # Initialize lists to track agents
        self.residents = []
        self.poi_agents = []  # List to store POI agents
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

        # Load parish-specific demographics if provided
        self.parish_demographics = kwargs.get('parish_demographics', {})
        
        # Load parishes GeoDataFrame if provided
        self.parishes_gdf = kwargs.get('parishes_gdf', None)
        
        # Create a mapping of nodes to parishes if parishes data is available
        self.node_to_parish = {}
        if self.parishes_gdf is not None:
            self._map_nodes_to_parishes()
        
        self.random = random.Random(kwargs.get('seed', None))
        
        # Set up scheduler and spatial environment
        self.schedule = CustomRandomActivation(self)
        self.space = GeoSpace()
        
        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Total Agents": lambda m: m.get_agent_count(),
                "Person Agents": lambda m: m.get_agent_count(agent_type=Resident),
                "POI Agents": lambda m: m.get_agent_count(agent_type=POI)
            },
            agent_reporters={
                "Position": lambda a: (a.geometry.x, a.geometry.y),
                "Type": lambda a: a.__class__.__name__,
                "Parish": lambda a: getattr(a, 'parish', None)
            }
        )
        
        # Create resident agents
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

            # Determine the parish this agent belongs to
            parish = self._get_parish_for_node(home_node)
            
            # Generate agent properties based on parish if available
            agent_props = self._generate_agent_properties(parish)

            resident = Resident(
                model=self,
                unique_id=i,
                geometry=point_geometry,
                home_node=home_node,
                accessible_nodes=accessible_nodes,
                parish=parish,
                **agent_props
            )
            self.grid.place_agent(resident, home_node)
            self.schedule.add(resident)  # Add to our custom scheduler
            self.residents.append(resident)
            self.all_agents.append(resident)

        # Create POI agents from the pois dictionary
        poi_id = num_residents  # Start POI IDs after resident IDs
        print("\nCreating POI agents from POI dictionary:")
        for category, poi_list in pois.items():
            print(f"Category: {category} - {len(poi_list)} POIs")
            for poi_data in poi_list:
                if isinstance(poi_data, tuple):
                    node_id, poi_type = poi_data  # Unpack node and type
                else:
                    node_id = poi_data  # If it's just a node ID
                    poi_type = category
                
                # Get coordinates from the node
                node_coords = self.graph.nodes[node_id]
                
                # Create a Point geometry from the coordinates
                point_geometry = Point(node_coords['x'], node_coords['y'])
                
                # Determine the parish this POI belongs to
                parish = self._get_parish_for_node(node_id)
                
                # Create the POI agent with explicit category
                poi_agent = POI(
                    model=self,
                    unique_id=poi_id,
                    geometry=point_geometry,
                    node_id=node_id,
                    poi_type=poi_type,
                    category=category,  # Explicitly pass the category
                    parish=parish
                )
                
                # Debug: Print the POI agent's category
                print(f"  Created POI {poi_id}: type={poi_type}, category={category}")
                
                self.grid.place_agent(poi_agent, node_id)
                self.schedule.add(poi_agent)
                self.poi_agents.append(poi_agent)
                self.all_agents.append(poi_agent)
                poi_id += 1

        self._initialize_social_networks(kwargs.get('social_network_density', 0.1))
        self.logger.info(f"Generated {num_residents} resident agents and {len(self.poi_agents)} POI agents")

    def _map_nodes_to_parishes(self):
        """
        Create a mapping from graph nodes to parishes.
        This allows quick lookup of which parish a node belongs to.
        """
        if self.parishes_gdf is None:
            self.logger.warning("No parishes data provided. Parish mapping not created.")
            return
            
        self.logger.info("Mapping network nodes to parishes...")
        nodes_count = len(self.graph.nodes())
        mapped_count = 0
        
        # Convert nodes to Points and check which parish they fall within
        for node_id, node_attrs in self.graph.nodes(data=True):
            if 'x' in node_attrs and 'y' in node_attrs:
                point = Point(node_attrs['x'], node_attrs['y'])
                
                # Check which parish contains this point
                for idx, parish in self.parishes_gdf.iterrows():
                    if parish.geometry.contains(point):
                        self.node_to_parish[node_id] = parish['name']
                        mapped_count += 1
                        break
        
        self.logger.info(f"Mapped {mapped_count} out of {nodes_count} nodes to parishes")
        
        # Check if we have too few mapped nodes
        if mapped_count < nodes_count * 0.5:  # If less than 50% mapped
            self.logger.warning("Many nodes could not be mapped to parishes. Check coordinate systems.")
    
    def _get_parish_for_node(self, node_id):
        """
        Get the parish name for a given node ID.
        
        Args:
            node_id: The ID of the node to look up
            
        Returns:
            String parish name or None if not found
        """
        return self.node_to_parish.get(node_id, None)
    
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
    
    def get_agents_by_parish(self, parish_name):
        """
        Get all agents in a specific parish.
        
        Args:
            parish_name: Name of the parish to filter by
            
        Returns:
            List of agents in the specified parish
        """
        return [agent for agent in self.all_agents if getattr(agent, 'parish', None) == parish_name]

    def step(self):
        """Advance the model by one step"""
        # Increment step counter
        self.step_count += 1
        
        # Advance time (each step is 1 hour)
        self.hour_of_day = (self.hour_of_day + 1) % 24
        if self.hour_of_day == 0:
            # New day
            self.day_of_week = (self.day_of_week + 1) % 7
            self.day_count += 1
            self.logger.info(f"Day {self.day_count}, {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][self.day_of_week]}")
        
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
    

    def _generate_agent_properties(self, parish=None):
        """
        Generate agent properties based on demographic distributions.
        If a parish is specified and parish-specific demographics exist,
        use those instead of the global demographics.
        
        Args:
            parish: The parish name to use for demographics (optional)
            
        Returns:
            Dictionary of agent properties
        """
        # Use parish-specific demographics if available
        demographics = self.demographics
        if parish and parish in self.parish_demographics:
            demographics = self.parish_demographics[parish]
        
        if not demographics:
            return {}
        
        props = {}
        
        # Generate age
        age_dist = demographics.get('age_distribution', {})
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
        gender_dist = demographics.get('gender_distribution', {})
        props['gender'] = self._sample_from_distribution(gender_dist)
        
        # Generate income
        income_dist = demographics.get('income_distribution', {})
        income_level = self._sample_from_distribution(income_dist)
        
        # Convert income level to actual income - use parish-specific ranges if available
        income_ranges = demographics.get('income_ranges', {
            "low": (10000, 30000),
            "medium": (30001, 100000),
            "high": (100001, 500000)
        })
        
        if income_level in income_ranges:
            min_val, max_val = income_ranges[income_level]
            props['income'] = self.random.randint(min_val, max_val)
        else:
            # Fallback to default ranges
            if income_level == "low":
                props['income'] = self.random.randint(10000, 30000)
            elif income_level == "medium":
                props['income'] = self.random.randint(30001, 100000)
            else:  # high
                props['income'] = self.random.randint(100001, 500000)
        
        # Generate education
        education_dist = demographics.get('education_distribution', {})
        props['education'] = self._sample_from_distribution(education_dist)
        
        # We no longer add parish to props since it's passed separately
        # This avoids the "got multiple values for keyword argument 'parish'" error
        
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