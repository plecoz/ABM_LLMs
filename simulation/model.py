import mesa
from mesa import Model
from mesa.space import NetworkGrid
import random
from agents.resident import Resident
from agents.poi import POI
import networkx as nx
import osmnx as ox
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
from outputs import OutputController

# Add imports for LLM integration
from agents.persona_memory_modules import PersonaMemoryManager, PersonaType
from simulation.llm_interaction_layer import LLMInteractionLayer

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
        # Print less frequently with 1-minute time steps - only every hour
        if self.steps % 60 == 0:
            print(f"Activating {len(self.agents)} agents at step {self.steps} (Hour {self.steps // 60})")
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
        """
        Initialize the model.
        
        Args:
            graph: NetworkX graph representing the street network
            pois: Dictionary of POIs by category
            num_residents: Number of resident agents to create
            **kwargs: Additional arguments:
                - parishes_gdf: GeoDataFrame with parish boundaries
                - parish_demographics: Dictionary of parish-specific demographics
                - parish_distribution: Dictionary mapping parishes to number of residents
                - random_distribution: Whether to distribute residents randomly
                - needs_selection: Method for generating resident needs ('random', 'maslow', 'capability', 'llms')
                - movement_behavior: Agent movement behavior ('need-based' or 'random')
                - seed: Random seed for reproducible results (default: 42)
        """
        # Get random seed from kwargs
        seed = kwargs.get('seed', 42)
        
        # Initialize Mesa's Model with the seed
        super().__init__(seed=seed)
        
        # Seed all random number generators consistently
        random.seed(seed)  # Python's random
        np.random.seed(seed)  # NumPy's random
        
        # Initialize logger
        self.logger = logging.getLogger("FifteenMinuteCity")
        self.logger.info(f"Initializing model with seed: {seed}")
        
        # Store city name
        self.city = kwargs.get('city', 'Macau, China')
        
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
        
        # Load residential buildings GeoDataFrame if provided
        self.residential_buildings = kwargs.get('residential_buildings', None)
        
        # Load environment data if provided
        self.water_bodies = kwargs.get('water_bodies', None)
        self.cliffs = kwargs.get('cliffs', None)
        self.forests = kwargs.get('forests', None)  # Green areas data
        
        # Get parish distribution and random distribution settings
        self.parish_distribution = kwargs.get('parish_distribution', None)
        self.random_distribution = kwargs.get('random_distribution', False)
        
        # Get needs selection method
        self.needs_selection = kwargs.get('needs_selection', 'random')
        
        # Get movement behavior setting
        self.movement_behavior = kwargs.get('movement_behavior', 'need-based')
        
        # Get action granularity setting (for POI activities)
        self.action_granularity = kwargs.get('action_granularity', 'basic')
        if isinstance(self.action_granularity, str):
            # Convert string to enum
            from agents.resident import ActionGranularity
            granularity_map = {
                'simple': ActionGranularity.SIMPLE,
                'basic': ActionGranularity.BASIC,
                'detailed': ActionGranularity.DETAILED
            }
            self.action_granularity = granularity_map.get(self.action_granularity.lower(), ActionGranularity.BASIC)
        
        # Initialize LLM components if needed
        self.llm_enabled = (self.needs_selection == 'llms' or self.movement_behavior == 'llms')
        if self.llm_enabled:
            self.logger.info("Initializing LLM components for persona-driven behavior")
            self.persona_memory_manager = PersonaMemoryManager()
            self.llm_interaction_layer = LLMInteractionLayer()
        else:
            self.persona_memory_manager = None
            self.llm_interaction_layer = None
        
        # Create a mapping of nodes to parishes if parishes data is available
        self.node_to_parish = {}
        if self.parishes_gdf is not None:
            self._map_nodes_to_parishes()
        
        self.random = random.Random(seed)
        #self.random = random.Random(0)
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
                "Parish": lambda a: getattr(a, 'parish', None),
                "Age": lambda a: getattr(a, 'age', None),
                "Age_Class": lambda a: getattr(a, 'age_class', None),
                "Income": lambda a: getattr(a, 'income', None),
                "Employment": lambda a: getattr(a, 'employment_status', None),
                "Household": lambda a: getattr(a, 'household_type', None),
                #"Energy": lambda a: getattr(a, 'energy', None),
                "Speed": lambda a: getattr(a, 'speed', None),
                "Traveling": lambda a: getattr(a, 'traveling', False),
                "Travel_Time_Remaining": lambda a: getattr(a, 'travel_time_remaining', 0)
            }
        )
        
        # Initialize output controller for tracking metrics
        self.output_controller = OutputController(self)
        
        # Create resident agents with proportional distribution
        self._create_residents_with_distribution(num_residents)

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

    def get_current_time(self):
        """
        Get the current simulation time in a readable format.
        
        Returns:
            Dictionary with current time information
        """
        minutes_elapsed = self.step_count
        total_hours = minutes_elapsed // 60
        current_minute = minutes_elapsed % 60
        
        # Calculate current hour and day
        hour = (8 + total_hours) % 24  # Start at 8 AM
        current_day = total_hours // 24
        day_of_week = current_day % 7
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return {
            'step': self.step_count,
            'minute': current_minute,
            'hour': hour,
            'day': current_day + 1,
            'day_of_week': day_names[day_of_week],
            'time_string': f"Day {current_day + 1} ({day_names[day_of_week]}) {hour:02d}:{current_minute:02d}"
        }

    def step(self):
        """Advance the model by one step"""
        # Increment step counter
        self.step_count += 1
        
        # Advance time (each step is 1 minute)
        minutes_elapsed = self.step_count
        total_hours = minutes_elapsed // 60
        current_minute = minutes_elapsed % 60
        
        # Calculate current hour and day
        self.hour_of_day = (8 + total_hours) % 24  # Start at 8 AM
        current_day = total_hours // 24
        self.day_of_week = current_day % 7
        self.day_count = current_day
        
        # Log new day transitions
        if self.step_count > 1 and current_minute == 0 and self.hour_of_day == 0:
            self.logger.info(f"Day {self.day_count + 1}, {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][self.day_of_week]}")
        
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

        # Default values for employment status and household type
        # These will be initialized later with sociodemographic data
        props['employment_status'] = "employed"
        props['household_type'] = "single"
        
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

    def _create_residents_with_distribution(self, num_residents):
        """
        Create resident agents with proportional distribution across parishes.
        If residential building data is available, residents are placed at building
        locations. Otherwise, they are placed at random nodes within the parish.
        
        Args:
            num_residents: Total number of residents to create
        """
        agent_id = 0
        
        # Check if we should use buildings for placement
        use_buildings = self.residential_buildings is not None and not self.residential_buildings.empty
        
        if use_buildings:
            self.logger.info("Initializing residents at residential building locations.")
        else:
            self.logger.info("No residential building data. Initializing residents at random network nodes.")

        for parish_name, num_parish_residents in self.parish_distribution.items():
            if num_parish_residents <= 0:
                continue
            
            # --- Home Location Selection ---
            home_locations = []
            
            if use_buildings and self.parishes_gdf is not None:
                # Get the geometry for the current parish
                parish_geom_series = self.parishes_gdf[self.parishes_gdf['name'].apply(
                    self._clean_parish_name_for_matching) == parish_name].geometry
                
                if not parish_geom_series.empty:
                    parish_geom = parish_geom_series.iloc[0]
                    # Find buildings within this parish
                    buildings_in_parish = self.residential_buildings[self.residential_buildings.within(parish_geom)]
                    
                    if not buildings_in_parish.empty:
                        # Sample buildings with replacement for the number of residents
                        selected_buildings = buildings_in_parish.sample(n=num_parish_residents, replace=True, random_state=self.random.randint(0, 1000000))
                        
                        for _, building in selected_buildings.iterrows():
                            # Use centroid for agent geometry
                            point_geometry = building.geometry.centroid
                            # Find nearest network node for travel
                            home_node = ox.distance.nearest_nodes(self.graph, point_geometry.x, point_geometry.y)
                            home_locations.append({'geometry': point_geometry, 'node': home_node})
                    else:
                        self.logger.warning(f"No residential buildings found in parish {parish_name}. Falling back to random nodes for this parish.")
                else:
                    self.logger.warning(f"Could not find geometry for parish {parish_name}. Falling back to random nodes.")
            
            # Fallback or default behavior: use random nodes
            if not home_locations:
                parish_nodes = [node_id for node_id, parish in self.node_to_parish.items() 
                              if parish and self._clean_parish_name_for_matching(parish) == parish_name]
                
                if not parish_nodes:
                    self.logger.warning(f"No network nodes found for parish {parish_name}. Skipping residents for this parish.")
                    continue
                
                for _ in range(num_parish_residents):
                    home_node = random.choice(parish_nodes)
                    node_coords = self.graph.nodes[home_node]
                    point_geometry = Point(node_coords['x'], node_coords['y'])
                    home_locations.append({'geometry': point_geometry, 'node': home_node})

            # --- Residents Creation ---
            for location in home_locations:
                home_node = location['node']
                point_geometry = location['geometry']
                
                # Calculate access distance from building centroid to nearest network node
                home_node_geom = self.graph.nodes[home_node]
                access_distance_meters = ox.distance.great_circle(
                    lat1=point_geometry.y, lon1=point_geometry.x,
                    lat2=home_node_geom['y'], lon2=home_node_geom['x']
                )
                print(f"DEBUG: Agent {agent_id} has an access distance of {access_distance_meters:.2f} meters.")
                parish = self._get_parish_for_node(home_node)
                agent_props = self._generate_agent_properties(parish)

                # Determine step size and 15-minute radius based on agent's age
                is_elderly = '65+' in agent_props.get('age_class', '') or agent_props.get('age', 0) >= 65
                step_size = 60.0 if is_elderly else 80.0
                fifteen_minute_radius = 15 * step_size

                accessible_nodes = dict(nx.single_source_dijkstra_path_length(
                    self.graph, home_node, cutoff=fifteen_minute_radius, weight='length'
                ))
                
                
                resident = Resident(
                    model=self, unique_id=agent_id, geometry=point_geometry,
                    home_node=home_node, accessible_nodes=accessible_nodes,
                    parish=parish, needs_selection=self.needs_selection,
                    movement_behavior=self.movement_behavior, 
                    access_distance=access_distance_meters,
                    **agent_props
                )
                
                # Assign persona if LLM behavior is enabled
                if self.llm_enabled:
                    self._assign_persona_to_resident(resident)
                
                self.grid.place_agent(resident, home_node)
                self.schedule.add(resident)
                self.residents.append(resident)
                self.all_agents.append(resident)
                agent_id += 1

    def _create_residents_randomly(self, num_residents):
        """
        Create residents with random distribution.
        If residential building data is available, residents are placed at random
        building locations. Otherwise, they are placed at random nodes.
        
        Args:
            num_residents: Total number of residents to create
        """
        home_locations = []
        use_buildings = self.residential_buildings is not None and not self.residential_buildings.empty

        if use_buildings:
            self.logger.info("Initializing residents at random residential building locations.")
            selected_buildings = self.residential_buildings.sample(n=num_residents, replace=True, random_state=self.random.randint(0, 1000000))
            for _, building in selected_buildings.iterrows():
                point_geometry = building.geometry.centroid
                home_node = ox.distance.nearest_nodes(self.graph, point_geometry.x, point_geometry.y)
                home_locations.append({'geometry': point_geometry, 'node': home_node})
        else:
            self.logger.info("No residential building data. Initializing residents at random network nodes.")
            random_nodes = random.choices(list(self.graph.nodes()), k=num_residents)
            for home_node in random_nodes:
                node_coords = self.graph.nodes[home_node]
                point_geometry = Point(node_coords['x'], node_coords['y'])
                home_locations.append({'geometry': point_geometry, 'node': home_node})

        for i, location in enumerate(home_locations):
            home_node = location['node']
            point_geometry = location['geometry']
            
            # Calculate access distance
            home_node_geom = self.graph.nodes[home_node]
            access_distance_meters = ox.distance.great_circle_vec(
                lat1=point_geometry.y, lon1=point_geometry.x,
                lat2=home_node_geom['y'], lon2=home_node_geom['x']
            )
            print(f"DEBUG: Agent {i} has an access distance of {access_distance_meters:.2f} meters.")
            
            agent_props = self._generate_agent_properties(parish)

            # Determine step size and 15-minute radius based on agent's age
            is_elderly = '65+' in agent_props.get('age_class', '') or agent_props.get('age', 0) >= 65
            step_size = 60.0 if is_elderly else 80.0
            fifteen_minute_radius = 15 * step_size

            accessible_nodes = dict(nx.single_source_dijkstra_path_length(
                self.graph, home_node, cutoff=fifteen_minute_radius, weight='length'
            ))
            parish = self._get_parish_for_node(home_node)
            
            resident = Resident(
                model=self, unique_id=i, geometry=point_geometry,
                home_node=home_node, accessible_nodes=accessible_nodes,
                parish=parish, needs_selection=self.needs_selection,
                movement_behavior=self.movement_behavior, 
                access_distance=access_distance_meters,
                **agent_props
            )
            
            # Assign persona if LLM behavior is enabled
            if self.llm_enabled:
                self._assign_persona_to_resident(resident)
            
            self.grid.place_agent(resident, home_node)
            self.schedule.add(resident)
            self.residents.append(resident)
            self.all_agents.append(resident)

    def _clean_parish_name_for_matching(self, parish_name):
        """
        Clean parish name for matching with distribution dictionary.
        This should match the cleaning function in main.py.
        
        Args:
            parish_name: Original parish name
            
        Returns:
            Cleaned parish name
        """
        if not parish_name:
            return parish_name
        
        import re
        import unicodedata
        
        # Remove Chinese characters (keep only Latin characters, numbers, spaces, and basic punctuation)
        cleaned = re.sub(r'[^\w\s\-\.\(\)]', '', parish_name, flags=re.ASCII)
        
        # Remove accents from letters using Unicode normalization
        normalized = unicodedata.normalize('NFD', cleaned)
        without_accents = ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')
        
        # Clean up extra spaces
        final_cleaned = ' '.join(without_accents.split())
        
        return final_cleaned.strip()

    def _assign_persona_to_resident(self, resident):
        """
        Assign a persona to a resident based on their demographic characteristics.
        
        Args:
            resident: The resident agent to assign a persona to
        """
        if not self.llm_enabled or not self.persona_memory_manager:
            return
        
        # Determine persona type based on resident characteristics
        persona_type = self._determine_persona_type(resident)
        
        # Create persona profile for the resident
        persona_template, emotional_state = self.persona_memory_manager.create_agent_persona(
            agent_id=str(resident.unique_id),
            persona_type=persona_type,
            variation_factor=0.1  # Add some variation to make agents unique
        )
        
        # Store persona information in the resident
        resident.persona_type = persona_type
        resident.persona_template = persona_template
        resident.emotional_state = emotional_state
        
        self.logger.debug(f"Assigned persona {persona_type.value} to resident {resident.unique_id}")
    
    def _determine_persona_type(self, resident):
        """
        Determine the appropriate persona type for a resident based on their characteristics.
        
        Args:
            resident: The resident agent
            
        Returns:
            PersonaType enum value
        """
        # Determine persona based on age and other characteristics
        age = getattr(resident, 'age', 30)
        employment_status = getattr(resident, 'employment_status', 'employed')
        
        # Age-based persona assignment with some randomness
        if age >= 65:
            return PersonaType.ELDERLY_RESIDENT
        elif age < 25:
            if employment_status == 'student':
                return PersonaType.STUDENT
            else:
                return PersonaType.YOUNG_PROFESSIONAL
        elif 25 <= age < 45:
            # For middle-aged adults, consider family status and employment
            household_type = getattr(resident, 'household_type', 'single')
            if 'family' in household_type.lower() or 'parent' in household_type.lower():
                return PersonaType.WORKING_PARENT
            else:
                return PersonaType.YOUNG_PROFESSIONAL
        else:  # 45-64
            # Could be working parent or professional
            if random.random() < 0.6:  # 60% chance of being working parent
                return PersonaType.WORKING_PARENT
            else:
                return PersonaType.YOUNG_PROFESSIONAL