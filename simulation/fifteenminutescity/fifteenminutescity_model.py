import mesa
from mesa import Model
from mesa.space import NetworkGrid
import random
from agents.fifteenminutescity.resident import Resident
from agents.fifteenminutescity.poi import POI
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
import os  # NEW

# Add imports for LLM integration
from agents.fifteenminutescity.persona_memory_modules import PersonaMemoryManager, PersonaType
from simulation.fifteenminutescity.llm_interaction_layer_fifteenminutescity import FifteenMinuteCityLLMLayer

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
                - threshold: Time threshold in minutes for accessibility (default: 15 for 15-minute city)
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
        
        # Store accessibility threshold (in minutes)
        self.threshold = kwargs.get('threshold', 15)
        self.logger.info(f"Using accessibility threshold: {self.threshold} minutes")
        
        self.graph = graph
        self.pois = pois
        self.grid = NetworkGrid(graph)
        
        # Add a step counter and interaction counter
        self.step_count = 0
        self.interactions_this_step = 0
        
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
            from agents.fifteenminutescity.resident import ActionGranularity
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
            self.llm_interaction_layer = FifteenMinuteCityLLMLayer()
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
                "POI Agents": lambda m: m.get_agent_count(agent_type=POI),
                "Interactions": lambda m: m.interactions_this_step
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

        # --- Load industry distribution data (education -> industry probabilities) ---  NEW BLOCK
        industry_path = kwargs.get('industry_path', 'data/demographics_macau/industry.json')
        self.industry_distribution = {}
        if self.city == "Macau, China" and industry_path and os.path.exists(industry_path):
            self._load_industry_distribution(industry_path)
        else:
            if self.city == "Macau, China":
                self.logger.warning(f"Industry distribution file not found at {industry_path}. 'industry' attribute will default to None.")

        # --- Load occupation distribution data (age -> occupation probabilities) ---
        occupation_path = kwargs.get('occupation_path', 'data/demographics_macau/occupation.json')
        self.occupation_distribution = {}
        if self.city == "Macau, China" and occupation_path and os.path.exists(occupation_path):
            self._load_occupation_distribution(occupation_path)
        else:
            if self.city == "Macau, China":
                self.logger.warning(f"Occupation distribution file not found at {occupation_path}. 'occupation' attribute will default to None.")

        # --- Load income distribution data (occupation -> income probabilities) ---
        income_path = kwargs.get('income_path', 'data/demographics_macau/income.json')
        self.income_distribution = {}
        if self.city == "Macau, China" and income_path and os.path.exists(income_path):
            self._load_income_distribution(income_path)
        else:
            if self.city == "Macau, China":
                self.logger.warning(f"Income distribution file not found at {income_path}. Income will be assigned using default ranges.")
        
        # --- Load parish-specific demographics if provided ---

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
        """
        return self.node_to_parish.get(node_id, None)
    
    def get_agent_by_id(self, agent_id):
        """Get an agent by its unique_id from the model's agent list."""
        # The CustomRandomActivation scheduler uses a simple list `agents`
        for agent in self.schedule.agents:
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
        # Reset per-step counters
        self.interactions_this_step = 0
        
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
    
    def _load_industry_distribution(self, industry_path):  # NEW METHOD
        """
        Load industry distribution probabilities from a JSON file.
        Expected format: {education_level: {industry_name: probability, ...}, ...}
        """
        try:
            with open(industry_path, 'r') as f:
                self.industry_distribution = json.load(f)
            # Normalise keys to facilitate matching (lowercase, stripped)
            self.industry_distribution = {
                self._normalise_string(k): v for k, v in self.industry_distribution.items()
            }
            self.logger.info("Loaded industry distribution data")
        except Exception as e:
            self.logger.error(f"Error loading industry distribution data: {e}")
            self.industry_distribution = {}

    def _load_occupation_distribution(self, occupation_path):
        """
        Load occupation distribution probabilities from a JSON file.
        Expected format: {age_group: {occupation_name: probability, ...}, ...}
        """
        try:
            with open(occupation_path, 'r') as f:
                self.occupation_distribution = json.load(f)
            self.logger.info("Loaded occupation distribution data")
        except Exception as e:
            self.logger.error(f"Error loading occupation distribution data: {e}")
            self.occupation_distribution = {}

    def _load_income_distribution(self, income_path):
        """
        Load income distribution probabilities from a JSON file.
        Expected format: {occupation: {income_level: probability, ...}, ...}
        """
        try:
            with open(income_path, 'r') as f:
                self.income_distribution = json.load(f)
            self.logger.info("Loaded income distribution data")
        except Exception as e:
            self.logger.error(f"Error loading income distribution data: {e}")
            self.income_distribution = {}

    def _normalise_string(self, s):  # NEW HELPER
        import re
        return re.sub(r"[^a-z0-9]", "", str(s).lower()) if s else ""

    def _convert_income_bracket_to_value(self, income_bracket):
        """
        Convert income bracket string to actual income value.
        
        Args:
            income_bracket: String like "20 000 - 29 999", "≧60 000", "< 3 500", etc.
            
        Returns:
            Random income value within the bracket range
        """
        if not income_bracket or income_bracket == "Unpaid family worker":
            return 0
        
        # Handle different bracket formats
        if "≧" in income_bracket:  # e.g., "≧60 000"
            min_income = int(income_bracket.replace("≧", "").replace(" ", ""))
            # For open-ended high brackets, use min + reasonable range
            return self.random.randint(min_income, min_income + 40000)
        elif "<" in income_bracket:  # e.g., "< 3 500"
            max_income = int(income_bracket.replace("<", "").replace(" ", ""))
            # For open-ended low brackets, use reasonable minimum
            return self.random.randint(max(1000, max_income - 1000), max_income - 1)
        elif "-" in income_bracket:  # e.g., "20 000 - 29 999"
            parts = income_bracket.split("-")
            if len(parts) == 2:
                min_income = int(parts[0].strip().replace(" ", ""))
                max_income = int(parts[1].strip().replace(" ", ""))
                return self.random.randint(min_income, max_income)
        
        # Fallback for unrecognized formats
        return None

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

        # --- AGE & AGE CLASS ---
        age_dist = demographics.get('age_distribution', {})
        age_group = self._sample_from_distribution(age_dist)
        props['age_class'] = age_group  # Store the sampled age class for reference

        # Convert age group to an actual age value
        if age_group is not None:
            if '-' in age_group:  # e.g. "25-29"
                min_age, max_age = age_group.split('-')
                try:
                    min_age = int(min_age)
                    max_age = int(max_age)
                    props['age'] = self.random.randint(min_age, max_age)
                except ValueError:
                    # Fallback in case parsing fails
                    props['age'] = self.random.randint(18, 90)
            elif age_group.endswith('+'):  # e.g. "85+"
                try:
                    min_age = int(age_group.rstrip('+'))
                    props['age'] = self.random.randint(min_age, min_age + 10)
                except ValueError:
                    props['age'] = self.random.randint(65, 90)
            else:  # Unknown pattern – fallback
                props['age'] = self.random.randint(18, 90)
        else:
            # If no age group could be sampled, fallback to a default range
            props['age'] = self.random.randint(18, 90)

        # --- GENDER ---
        gender_dist = {}
        # Prefer age-specific gender distribution if available
        if demographics.get('gender_by_age') and age_group in demographics['gender_by_age']:
            gender_dist = demographics['gender_by_age'][age_group]
        else:
            gender_dist = demographics.get('gender_distribution', {})
        props['gender'] = self._sample_from_distribution(gender_dist)

        # --- INCOME ---
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

        # Note: For Macau, income will be reassigned based on occupation after occupation is determined

        # --- EDUCATION LEVEL ---
        education_level = None
        if (demographics.get('education_distribution_by_age_and_gender') and
            age_group in demographics['education_distribution_by_age_and_gender'] and
            props['gender'] in demographics['education_distribution_by_age_and_gender'][age_group]):
            edu_dist_raw = demographics['education_distribution_by_age_and_gender'][age_group][props['gender']]
            # Flatten nested structures (e.g. Primary education -> complete/incomplete)
            flattened_edu_dist = {}
            for k, v in edu_dist_raw.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flattened_edu_dist[f"{k} {sub_k}"] = sub_v
                else:
                    flattened_edu_dist[k] = v
            education_level = self._sample_from_distribution(flattened_edu_dist)
        else:
            # Fallback to generic education distribution if provided
            education_dist = demographics.get('education_distribution', {})
            education_level = self._sample_from_distribution(education_dist)
        props['education'] = education_level

        # --- INDUSTRY ---  NEW SECTION
        industry_choice = None
        if (self.city == "Macau, China" and education_level 
            and getattr(self, 'industry_distribution', {})):
            norm_edu = self._normalise_string(education_level)
            # direct match or attempt to map common synonyms
            industry_dist = self.industry_distribution.get(norm_edu)
            if not industry_dist:
                # Try some heuristic replacements (e.g., replace 'primaryeducationcomplete' with 'primaryeducationcomplete') already same
                for key, dist in self.industry_distribution.items():
                    if key in norm_edu or norm_edu in key:
                        industry_dist = dist
                        break
            if industry_dist:
                industry_choice = self._sample_from_distribution(industry_dist)
        props['industry'] = industry_choice

        # --- OCCUPATION ---
        occupation_choice = None
        if (self.city == "Macau, China" and props.get('age') is not None 
            and getattr(self, 'occupation_distribution', {})):
            age = props['age']
            # Map age to age group used in occupation.json
            age_group = None
            if 16 <= age <= 24:
                age_group = "16-24"
            elif 25 <= age <= 29:
                age_group = "25-29"
            elif 30 <= age <= 34:
                age_group = "30-34"
            elif 35 <= age <= 39:
                age_group = "35-39"
            elif 40 <= age <= 44:
                age_group = "40-44"
            elif 45 <= age <= 49:
                age_group = "45-49"
            elif 50 <= age <= 54:
                age_group = "50-54"
            elif 55 <= age <= 59:
                age_group = "55-59"
            elif 60 <= age <= 64:
                age_group = "60-64"
            elif age >= 65:
                age_group = "≧65"
            
            if age_group and age_group in self.occupation_distribution:
                occupation_dist = self.occupation_distribution[age_group]
                occupation_choice = self._sample_from_distribution(occupation_dist)
        props['occupation'] = occupation_choice

        # --- INCOME (Reassign for Macau based on occupation) ---
        if (self.city == "Macau, China" and occupation_choice 
            and getattr(self, 'income_distribution', {})):
            # Handle slight naming differences between occupation.json and income.json
            income_key = occupation_choice
            
            if income_key in self.income_distribution:
                income_bracket_dist = self.income_distribution[income_key]
                income_bracket = self._sample_from_distribution(income_bracket_dist)
                
                # Convert income bracket to actual income value
                if income_bracket:
                    actual_income = self._convert_income_bracket_to_value(income_bracket)
                    if actual_income is not None:
                        props['income'] = actual_income

        # Default values for additional attributes
        props['employment_status'] = "employed"
        props['household_type'] = "single"

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
        # Handle case where no residents are requested (for visualization-only mode)
        if num_residents <= 0:
            self.logger.info("No residents requested. Skipping resident creation.")
            return
        
        # Handle case where parish_distribution is None (fallback to random distribution)
        if self.parish_distribution is None:
            self.logger.info("No parish distribution provided. Using random distribution.")
            self._create_residents_randomly(num_residents)
            return
            
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
                        # Calculate building areas for proportional sampling using projected coordinates
                        buildings_in_parish = self._calculate_building_areas(buildings_in_parish)
                        
                        # Remove buildings with zero or negative area
                        buildings_in_parish = buildings_in_parish[buildings_in_parish['area'] > 0]
                        
                        if not buildings_in_parish.empty:
                            # Sample buildings proportionally to their area
                            weights = buildings_in_parish['area'] / buildings_in_parish['area'].sum()
                            selected_buildings = buildings_in_parish.sample(
                                n=num_parish_residents, 
                                replace=True, 
                                weights=weights,
                                random_state=self.random.randint(0, 1000000)
                            )
                            
                            self.logger.info(f"Parish {parish_name}: Selected {num_parish_residents} residents from {len(buildings_in_parish)} buildings (area-weighted)")
                            
                            for _, building in selected_buildings.iterrows():
                                # Use centroid for agent geometry
                                point_geometry = building.geometry.centroid
                                # Find nearest network node for travel
                                home_node = ox.distance.nearest_nodes(self.graph, point_geometry.x, point_geometry.y)
                                home_locations.append({'geometry': point_geometry, 'node': home_node})
                        else:
                            self.logger.warning(f"No residential buildings with valid area found in parish {parish_name}. Falling back to random nodes for this parish.")
                    else:
                        self.logger.warning(f"No residential buildings found in parish {parish_name}. Falling back to random nodes for this parish.")
            
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
            for i, location in enumerate(home_locations):
                home_node = location['node']
                point_geometry = location['geometry']
                
                # Calculate access distance from building centroid to nearest network node
                home_node_geom = self.graph.nodes[home_node]
                access_distance_meters = ox.distance.great_circle(
                    lat1=point_geometry.y, lon1=point_geometry.x,
                    lat2=home_node_geom['y'], lon2=home_node_geom['x']
                )
                parish = self._get_parish_for_node(home_node)
                agent_props = self._generate_agent_properties(parish)
                
                # TEMPORARY FEATURE: For Taipa parish, spawn 30% of residents at casinos
                if parish_name == "Taipa" and self.pois.get('casino') and i < len(home_locations) * 0.3:
                    # Find a random casino location
                    casino_pois = self.pois['casino']
                    if casino_pois:
                        casino_node, _ = self.random.choice(casino_pois)
                        casino_coords = self.graph.nodes[casino_node]
                        point_geometry = Point(casino_coords['x'], casino_coords['y'])
                        home_node = casino_node
                        # TEMPORARY: Mark as tourist for special visualization
                        agent_props['is_tourist'] = True
                        self.logger.info(f"Spawning Taipa resident {agent_id} at casino location as tourist")

                # Determine step size and accessibility radius based on agent's age
                is_elderly = '65+' in agent_props.get('age_class', '') or agent_props.get('age', 0) >= 65
                step_size = 60.0 if is_elderly else 80.0
                accessibility_radius = self.threshold * step_size

                accessible_nodes = dict(nx.single_source_dijkstra_path_length(
                    self.graph, home_node, cutoff=accessibility_radius, weight='length'
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
        # Handle case where no residents are requested (for visualization-only mode)
        if num_residents <= 0:
            self.logger.info("No residents requested. Skipping resident creation.")
            return
            
        home_locations = []
        use_buildings = self.residential_buildings is not None and not self.residential_buildings.empty

        if use_buildings:
            self.logger.info("Initializing residents at random residential building locations.")
            
            # Calculate building areas for proportional sampling using projected coordinates
            residential_buildings_copy = self._calculate_building_areas(self.residential_buildings)
            
            # Remove buildings with zero or negative area
            residential_buildings_copy = residential_buildings_copy[residential_buildings_copy['area'] > 0]
            
            if not residential_buildings_copy.empty:
                # Sample buildings proportionally to their area
                weights = residential_buildings_copy['area'] / residential_buildings_copy['area'].sum()
                selected_buildings = residential_buildings_copy.sample(
                    n=num_residents, 
                    replace=True, 
                    weights=weights,
                    random_state=self.random.randint(0, 1000000)
                )
                
                self.logger.info(f"Selected {num_residents} residents from {len(residential_buildings_copy)} buildings (area-weighted)")
                
                for _, building in selected_buildings.iterrows():
                    point_geometry = building.geometry.centroid
                    home_node = ox.distance.nearest_nodes(self.graph, point_geometry.x, point_geometry.y)
                    home_locations.append({'geometry': point_geometry, 'node': home_node})
            else:
                self.logger.warning("No residential buildings with valid area found. Falling back to random network nodes.")
                use_buildings = False
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
            # print(f"DEBUG: Agent {i} has an access distance of {access_distance_meters:.2f} meters.")
            
            agent_props = self._generate_agent_properties(parish)

            # Determine step size and accessibility radius based on agent's age
            is_elderly = '65+' in agent_props.get('age_class', '') or agent_props.get('age', 0) >= 65
            step_size = 60.0 if is_elderly else 80.0
            accessibility_radius = self.threshold * step_size

            accessible_nodes = dict(nx.single_source_dijkstra_path_length(
                self.graph, home_node, cutoff=accessibility_radius, weight='length'
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

    def _calculate_building_areas(self, buildings_gdf):
        """
        Calculate building areas using proper projected coordinates to avoid geographic CRS warnings.
        
        Args:
            buildings_gdf: GeoDataFrame with building geometries
            
        Returns:
            GeoDataFrame with 'area' column added using projected coordinates
        """
        if buildings_gdf.empty:
            return buildings_gdf
        
        # Make a copy to avoid modifying the original
        buildings_copy = buildings_gdf.copy()
        
        # Check if we're in a geographic CRS (lat/lon)
        if buildings_copy.crs and buildings_copy.crs.is_geographic:
            # For Macau, use UTM Zone 49N (EPSG:32649) which is appropriate for the region
            # For other cities, we could use a more general approach like Web Mercator (EPSG:3857)
            if 'Macau' in self.city:
                projected_crs = 'EPSG:32649'  # UTM Zone 49N for Macau
            elif 'Barcelona' in self.city:
                projected_crs = 'EPSG:32631'  # UTM Zone 31N for Barcelona
            elif 'Hong Kong' in self.city:
                projected_crs = 'EPSG:32650'  # UTM Zone 50N for Hong Kong
            else:
                # Default to Web Mercator for other cities
                projected_crs = 'EPSG:3857'
            
            # Project to appropriate UTM zone and calculate area
            buildings_projected = buildings_copy.to_crs(projected_crs)
            buildings_copy['area'] = buildings_projected.geometry.area
            
            self.logger.debug(f"Calculated building areas using projected CRS: {projected_crs}")
        else:
            # Already in projected coordinates, calculate area directly
            buildings_copy['area'] = buildings_copy.geometry.area
            self.logger.debug("Calculated building areas using existing projected CRS")
        
        return buildings_copy