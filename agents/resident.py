from mesa.agent import Agent
from agents.base_agent import BaseAgent
import random
import networkx as nx
import logging
import math

class Resident(BaseAgent):
    def __init__(self, model, unique_id, geometry, home_node, accessible_nodes, **kwargs):
        """
        Initialize a resident agent.
        
        Args:
            model: Model instance the agent belongs to
            unique_id: Unique identifier for the agent
            geometry: Shapely geometry object representing the agent's location
            home_node: The agent's home node in the network
            accessible_nodes: Dictionary of nodes accessible to the agent
            **kwargs: Additional agent properties that can be customized
        """
        # Pass parameters in the correct order to parent class
        super().__init__(model, unique_id, geometry, **kwargs)
        
        # Custom attributes
        self.home_node = home_node
        self.current_node = home_node
        self.accessible_nodes = accessible_nodes
        self.visited_pois = []
        self.mobility_mode = "walk"
        # Person-specific attributes
        self.family_id = kwargs.get('family_id', None)
        self.household_members = kwargs.get('household_members', [])
        self.social_network = kwargs.get('social_network', [])
        self.daily_schedule = kwargs.get('daily_schedule', {})
        self.personality_traits = kwargs.get('personality_traits', {})
        self.activity_preferences = kwargs.get('activity_preferences', {})
        
        # Parish information
        self.parish = kwargs.get('parish', None)
        
        # Demographic attributes from model (may vary by parish)
        self.age = kwargs.get('age', 30)  # Default age
        self.gender = kwargs.get('gender', 'male')  # Default gender
        self.income = kwargs.get('income', 50000)  # Default income
        self.education = kwargs.get('education', 'high_school')  # Default education level
        
        # New attributes
        # Using default values for employment status and household type
        self.employment_status = kwargs.get('employment_status', "employed")
        self.household_type = kwargs.get('household_type', "single")
        
        # Dynamic needs (placeholder - to be implemented later)
        self.dynamic_needs = {
            "hunger": 0,
            "social": 0,
            "recreation": 0,
            "shopping": 0,
            "healthcare": 0,
            "education": 0
        }
        
        # Energy levels and mobility constraints
        self.max_energy = 100
        self.energy = self.max_energy
        # Age-based energy depletion rate: older agents lose energy faster
        if self.age < 18:
            self.energy_depletion_rate = 2  # Children have moderate depletion
        elif self.age < 35:
            self.energy_depletion_rate = 1  # Young adults have lowest depletion
        elif self.age < 65:
            self.energy_depletion_rate = 2  # Middle-aged adults have moderate depletion
        else:
            self.energy_depletion_rate = 3  # Elderly have highest depletion
        
        # Recharge counter when at home
        self.home_recharge_counter = 0
        
        # Mobility constraints - speed in km/h
        if self.age >= 65:
            self.speed = 3.0  # Elderly walk at 3 km/h
        else:
            self.speed = 5.0  # Everyone else walks at 5 km/h
        
        # Travel time tracking
        self.traveling = False
        self.travel_time_remaining = 0
        self.destination_node = None
        self.destination_geometry = None
        
        # Memory module
        self.memory = {
            'income': self.income,
            'visited_pois': []  # List of dicts: {step, poi_id, poi_type, category, income}
        }
        
        # Initialize logger if not provided
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"Resident-{unique_id}")

    def calculate_travel_time(self, from_node, to_node):
        """
        Calculate the travel time between two nodes based on distance and agent's speed.
        
        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            
        Returns:
            Number of time steps needed for travel (each step is 15 minutes)
        """
        # Get the distance in meters from the accessible_nodes dictionary
        # or calculate using shortest path if not directly accessible
        if to_node in self.accessible_nodes:
            distance_meters = self.accessible_nodes[to_node]
        else:
            try:
                # Calculate shortest path length
                distance_meters = nx.shortest_path_length(
                    self.model.graph, 
                    from_node, 
                    to_node, 
                    weight='length'
                )
            except (nx.NetworkXNoPath, KeyError):
                self.logger.warning(f"No path found from {from_node} to {to_node}")
                return None

        # Convert distance from meters to kilometers
        distance_km = distance_meters / 1000
        
        # Calculate travel time in hours
        travel_time_hours = distance_km / self.speed
        
        # Convert to minutes
        travel_time_minutes = travel_time_hours * 60
        
        # Convert to number of time steps (15-minute steps)
        time_steps_exact = travel_time_minutes / 15
        
        # Round according to the specified rule
        time_steps_floor = math.floor(time_steps_exact)
        remainder_minutes = (time_steps_exact - time_steps_floor) * 15
        
        if remainder_minutes < 8:
            time_steps = time_steps_floor
        else:
            time_steps = math.ceil(time_steps_exact)
        
        # Ensure at least 1 time step
        return max(1, time_steps)

    def start_travel(self, target_node, target_geometry):
        """
        Start traveling to a target node.
        
        Args:
            target_node: Destination node ID
            target_geometry: Destination geometry
            
        Returns:
            Boolean indicating if travel was successfully started
        """
        travel_time = self.calculate_travel_time(self.current_node, target_node)
        
        if travel_time is None:
            return False
        
        self.traveling = True
        self.travel_time_remaining = travel_time
        self.destination_node = target_node
        self.destination_geometry = target_geometry
        
        return True

    def move_to_poi(self, poi_type):
        """
        Move to a POI of the specified type.
        
        This updated version first tries to find POI agents of the specified type,
        and if found, moves to one of them. If no POI agents are found, falls back
        to the original behavior of using POI nodes.
        
        Args:
            poi_type: The type of POI to move to
            
        Returns:
            Boolean indicating if the move was successful
        """
        # If already traveling, don't start a new journey
        if self.traveling:
            return False
            
        # First try to find POI agents of the specified type
        poi_agents = [agent for agent in self.model.poi_agents if agent.poi_type == poi_type]
        
        if poi_agents:
            # Filter POIs to only those at accessible nodes
            accessible_pois = [poi for poi in poi_agents if poi.node_id in self.accessible_nodes]
            
            if accessible_pois:
                # Choose a random accessible POI
                target_poi = random.choice(accessible_pois)
                
                # Start traveling to the POI
                return self.start_travel(target_poi.node_id, target_poi.geometry)
        
        # Fall back to original behavior if no POI agents are found
        if not self.model.pois.get(poi_type):
            return False
        
        valid_pois = []
        for poi_entry in self.model.pois[poi_type]:
            if isinstance(poi_entry, tuple):
                node_id, _ = poi_entry  # Unpack node and type
            else:
                node_id = poi_entry  # If it's just a node ID
                
            if node_id in self.accessible_nodes:
                valid_pois.append(node_id)
        
        if not valid_pois:
            return False
            
        try:
            target = random.choice(valid_pois)
            
            # Get target node coordinates for geometry
            node_coords = self.model.graph.nodes[target]
            target_geometry = None
            if 'x' in node_coords and 'y' in node_coords:
                from shapely.geometry import Point
                target_geometry = Point(node_coords['x'], node_coords['y'])
            
            # Start traveling to the POI
            return self.start_travel(target, target_geometry)
                
        except (nx.NetworkXNoPath, KeyError, IndexError) as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error moving to POI: {e}")
        return False

    def step(self):
        """Advance the agent one step"""
        try:
            super().step()
            
            # Update energy levels
            if self.current_node != self.home_node:
                # Deplete energy when not at home
                self.energy = max(0, self.energy - self.energy_depletion_rate)
                self.home_recharge_counter = 0
            else:
                # At home - reset recharge counter or recharge energy
                if self.energy < self.max_energy:
                    self.home_recharge_counter += 1
                    # Recharge after staying home for 2 time steps
                    if self.home_recharge_counter >= 2:
                        self.energy = self.max_energy
                        self.home_recharge_counter = 0
            
            # Check energy level - if depleted, go home immediately
            if self.energy <= 0 and self.current_node != self.home_node:
                # Cancel any ongoing travel
                self.traveling = False
                self.travel_time_remaining = 0
                
                # Go straight home
                self.current_node = self.home_node
                node_coords = self.model.graph.nodes[self.home_node]
                if 'x' in node_coords and 'y' in node_coords:
                    from shapely.geometry import Point
                    self.geometry = Point(node_coords['x'], node_coords['y'])
                return  # Skip regular movement behavior
            
            # Handle ongoing travel
            if self.traveling:
                self.travel_time_remaining -= 1
                
                # Check if we've arrived
                if self.travel_time_remaining <= 0:
                    self.traveling = False
                    self.current_node = self.destination_node
                    self.geometry = self.destination_geometry
                    self.visited_pois.append(self.destination_node)
                    
                    # Add resident to POI's visitors if it's a POI agent
                    for poi in self.model.poi_agents:
                        if poi.node_id == self.destination_node and hasattr(poi, 'visitors'):
                            poi.visitors.add(self.unique_id)
                            # --- MEMORY MODULE: Record visit ---
                            self.memory['visited_pois'].append({
                                'step': getattr(self.model, 'step_count', None),
                                'poi_id': poi.unique_id,
                                'poi_type': poi.poi_type,
                                'category': getattr(poi, 'category', None),
                                'income': self.income
                            })
                    
                    # Reset travel attributes
                    self.destination_node = None
                    self.destination_geometry = None
                
                # Still traveling, don't take any other movement actions
                return
            
            # Regular movement behavior - only if energy is not depleted and not traveling
            if self.energy > 0 and not self.traveling:
                # Try to move to a POI agent
                if hasattr(self.model, 'poi_agents') and self.model.poi_agents:
                    # Get all unique POI types
                    poi_types = list(set(poi.poi_type for poi in self.model.poi_agents))
                    if poi_types:
                        self.move_to_poi(random.choice(poi_types))
                # Fall back to old method if POI agents aren't available
                elif hasattr(self.model, 'pois') and self.model.pois:
                    self.move_to_poi(random.choice(list(self.model.pois.keys())))
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in resident step: {e}")
    
    def set_activity_preferences(self, preferences):
        """
        Update the agent's activity preferences.
        
        Args:
            preferences: Dictionary of activity types and their weights
        """
        self.activity_preferences = preferences
    
    def add_to_social_network(self, agent_id):
        """
        Add an agent to this agent's social network.
        
        Args:
            agent_id: ID of the agent to add
        """
        if agent_id != self.unique_id and agent_id not in self.social_network:
            self.social_network.append(agent_id)
    
    def remove_from_social_network(self, agent_id):
        """
        Remove an agent from this agent's social network.
        
        Args:
            agent_id: ID of the agent to remove
        """
        if agent_id in self.social_network:
            self.social_network.remove(agent_id)
            
    def get_parish_info(self):
        """
        Get information about the agent's parish.
        
        Returns:
            Dictionary with parish details or None if no parish assigned
        """
        if not self.parish:
            return None
            
        return {
            "parish_name": self.parish,
            "home_location": (self.geometry.x, self.geometry.y),
            "demographic_info": {
                "age": self.age,
                "gender": self.gender,
                "income": self.income,
                "education": self.education,
                "employment_status": self.employment_status,
                "household_type": self.household_type
            }
        }

    def get_memory(self):
        """Return the resident's memory (income and visited POIs)."""
        return self.memory
