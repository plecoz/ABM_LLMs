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
        # Track the last visited node to prevent consecutive visits to same POI
        self.last_visited_node = None
        
        # Person-specific attributes
        self.family_id = kwargs.get('family_id', None)
        self.household_members = kwargs.get('household_members', [])
        self.social_network = kwargs.get('social_network', [])
        self.daily_schedule = kwargs.get('daily_schedule', {})
        self.personality_traits = kwargs.get('personality_traits', {})
        self.activity_preferences = kwargs.get('activity_preferences', {})
        
        # Parish information
        self.parish = kwargs.get('parish', None)
        
        # Needs selection method
        self.needs_selection = kwargs.get('needs_selection', 'random')
        
        # Movement behavior setting
        self.movement_behavior = kwargs.get('movement_behavior', 'need-based')
        

        # Dynamic needs (placeholder - to be implemented later)
        self.dynamic_needs = {
            "hunger": 0,
            "social": 0,
            "recreation": 0,
            "shopping": 0,
            "healthcare": 0,
            "education": 0
        }

        # Current needs (will be updated each step)
        self.current_needs = self.dynamic_needs.copy()
        
        # Demographic attributes from model (may vary by parish)
        self.age = kwargs.get('age', 30)  # Default age
        self.gender = kwargs.get('gender', 'male')  # Default gender
        self.income = kwargs.get('income', 50000)  # Default income
        self.education = kwargs.get('education', 'high_school')  # Default education level
        
        # New attributes
        # Using default values for employment status and household type
        self.employment_status = kwargs.get('employment_status', "employed")
        self.household_type = kwargs.get('household_type', "single")
        

        
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
        
        # Waiting time tracking (new)
        self.waiting_at_poi = False
        self.waiting_time_remaining = 0
        
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
        Calculate the travel time between two nodes based on 80-meter steps.
        Each step represents 80 meters of walking distance (1 minute at 5km/h).
        Always rounds up to ensure the agent doesn't move more than 80m per step.
        
        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            
        Returns:
            Number of time steps needed for travel (each step is 1 minute = 80 meters)
        """
        # Always calculate the actual shortest path length for consistency
        try:
            # Calculate shortest path length along the street network
            distance_meters = nx.shortest_path_length(
                self.model.graph, 
                from_node, 
                to_node, 
                weight='length'
            )
        except (nx.NetworkXNoPath, KeyError):
            self.logger.warning(f"No path found from {from_node} to {to_node}")
            return None

        # Calculate number of 80-meter steps needed
        # Always round UP to ensure no step exceeds 80 meters
        steps_needed = math.ceil(distance_meters / 80.0)
        
        # Debug output for resident 0
        if hasattr(self, 'unique_id') and self.unique_id == 0:
            print(f"Resident 0: Distance {distance_meters:.1f}m → {steps_needed} steps (80m each)")
        
        # Ensure at least 1 time step
        return max(1, steps_needed)

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
        
        # Notify output controller about travel start
        if hasattr(self.model, 'output_controller'):
            self.model.output_controller.track_travel_start(
                agent_id=self.unique_id,
                from_node=self.current_node,
                to_node=target_node,
                travel_time=travel_time
            )
        
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

    def get_need_to_poi_mapping(self):
        """
        Map needs to POI types that can satisfy them.
        
        Returns:
            Dictionary mapping need types to lists of POI types
        """
        return {
            "hunger": ["restaurant", "cafe", "fast_food", "food_court", "bar", "pub"],
            "social": ["restaurant", "cafe", "bar", "pub", "community_centre", "place_of_worship", "park"],
            "recreation": ["park", "cinema", "theatre", "sports_centre", "museum", "library", "tourist_attraction"],
            "shopping": ["shop", "supermarket", "mall", "marketplace", "department_store"],
            "healthcare": ["hospital", "clinic", "pharmacy", "dentist", "doctor"],
            "education": ["school", "university", "college", "library", "training_centre"]
        }

    def find_highest_need(self):
        """
        Find the need with the highest value.
        
        Returns:
            Tuple of (need_type, need_value) for the highest need
        """
        if not self.current_needs:
            return None, 0
        
        highest_need = max(self.current_needs.items(), key=lambda x: x[1])
        return highest_need

    def find_poi_for_need(self, need_type):
        """
        Find available POI types that can satisfy a specific need.
        
        Args:
            need_type: The type of need to satisfy
            
        Returns:
            List of available POI types that can satisfy the need
        """
        need_mapping = self.get_need_to_poi_mapping()
        possible_poi_types = need_mapping.get(need_type, [])
        
        # Filter to only POI types that actually exist in the model
        available_poi_types = []
        
        # Check POI agents first
        if hasattr(self.model, 'poi_agents') and self.model.poi_agents:
            existing_poi_types = set(poi.poi_type for poi in self.model.poi_agents)
            available_poi_types.extend([poi_type for poi_type in possible_poi_types if poi_type in existing_poi_types])
        
        # Fall back to POI dictionary if no POI agents
        if not available_poi_types and hasattr(self.model, 'pois') and self.model.pois:
            existing_poi_types = set(self.model.pois.keys())
            available_poi_types.extend([poi_type for poi_type in possible_poi_types if poi_type in existing_poi_types])
        
        return available_poi_types

    def satisfy_need_at_poi(self, need_type, satisfaction_amount=30):
        """
        Reduce a need when visiting a POI that satisfies it.
        
        Args:
            need_type: The type of need being satisfied
            satisfaction_amount: How much the need is reduced (default: 30)
        """
        if need_type in self.current_needs:
            self.current_needs[need_type] = max(0, self.current_needs[need_type] - satisfaction_amount)

    def choose_movement_target(self):
        """
        Choose where to move based on movement behavior setting.
        For random movement, if no suitable POI is available (to avoid consecutive visits),
        the agent will be forced to go home.
        
        Returns:
            POI type to move to, 'home' to go home, or None if no movement should occur
        """
        if self.movement_behavior == 'need-based':
            return self._choose_need_based_target()
        else:  # random movement
            target = self._choose_random_target()
            if target is None and self.current_node != self.home_node:
                # No suitable POI available and not at home - force going home
                return 'home'
            return target

    def _choose_need_based_target(self):
        """
        Choose movement target based on current needs.
        
        Returns:
            POI type to move to, or None if no suitable POI found
        """
        # Update current needs
        self.current_needs = self.generate_needs()
        
        # Find the highest need
        highest_need_type, highest_need_value = self.find_highest_need()
        
        # Only move if the need is above a threshold (e.g., 50)
        if highest_need_value < 50:
            return None
        
        # Find POI types that can satisfy this need
        available_poi_types = self.find_poi_for_need(highest_need_type)
        
        if available_poi_types:
            # Choose a random POI type from available options
            return random.choice(available_poi_types)
        
        return None

    def _choose_random_target(self):
        """
        Choose a random POI type for movement.
        For random movement, prevents visiting the same POI twice in a row.
        If no other POIs are available, returns None to trigger going home.
        
        Returns:
            Random POI type, or None if no suitable POIs available
        """
        # Get all available POI types and their nodes
        available_poi_options = []
        
        if hasattr(self.model, 'poi_agents') and self.model.poi_agents:
            # Get POI agents and their node locations
            for poi in self.model.poi_agents:
                if poi.node_id in self.accessible_nodes:
                    # Skip if this is the last visited node (prevent consecutive visits)
                    if self.last_visited_node is not None and poi.node_id == self.last_visited_node:
                        continue
                    available_poi_options.append(poi.poi_type)
        elif hasattr(self.model, 'pois') and self.model.pois:
            # Fall back to POI dictionary
            for poi_type, poi_list in self.model.pois.items():
                for poi_entry in poi_list:
                    if isinstance(poi_entry, tuple):
                        node_id, _ = poi_entry
                    else:
                        node_id = poi_entry
                    
                    if node_id in self.accessible_nodes:
                        # Skip if this is the last visited node (prevent consecutive visits)
                        if self.last_visited_node is not None and node_id == self.last_visited_node:
                            continue
                        available_poi_options.append(poi_type)
        
        # Remove duplicates while preserving order
        available_poi_types = list(dict.fromkeys(available_poi_options))
        
        if available_poi_types:
            return random.choice(available_poi_types)
        
        # No suitable POIs available - return None to trigger going home
        return None

    def _satisfy_needs_at_poi(self, poi_type):
        """
        Satisfy needs when visiting a POI of a specific type.
        
        Args:
            poi_type: The type of POI being visited
        """
        need_mapping = self.get_need_to_poi_mapping()
        
        # Find which needs this POI type can satisfy
        for need_type, poi_types in need_mapping.items():
            if poi_type in poi_types:
                # Satisfy this need
                satisfaction_amount = random.randint(20, 40)  # Random satisfaction between 20-40
                self.satisfy_need_at_poi(need_type, satisfaction_amount)

    def increase_needs_over_time(self):
        """
        Gradually increase needs over time to simulate natural need accumulation.
        With 1-minute time steps, needs increase more slowly.
        """
        # Only increase needs every 15 minutes (every 15 steps) to avoid too rapid changes
        if self.model.step_count % 15 == 0:
            for need_type in self.current_needs:
                # Increase each need by a small random amount (1-3 points per 15 minutes)
                increase = random.randint(1, 3)
                self.current_needs[need_type] = min(100, self.current_needs[need_type] + increase)

    def step(self):
        """Advance the agent one step"""
        try:
            super().step()
            
            # Increase needs over time
            self.increase_needs_over_time()
            
            # Update energy levels
            if self.current_node != self.home_node:
                # Deplete energy when not at home - only every 30 minutes to be more realistic
                if self.model.step_count % 30 == 0:
                    self.energy = max(0, self.energy - self.energy_depletion_rate)
                self.home_recharge_counter = 0
            else:
                # At home - reset recharge counter or recharge energy
                if self.energy < self.max_energy:
                    self.home_recharge_counter += 1
                    # Recharge after staying home for 120 minutes (2 hours)
                    if self.home_recharge_counter >= 120:
                        self.energy = self.max_energy
                        self.home_recharge_counter = 0
            
            # Check energy level - if depleted, go home immediately
            if self.energy <= 0 and self.current_node != self.home_node:
                # Cancel any ongoing travel
                self.traveling = False
                self.travel_time_remaining = 0
                
                # Update last visited node before going home
                self.last_visited_node = self.current_node
                
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
                    # Update last visited node before changing current node
                    self.last_visited_node = self.current_node
                    self.current_node = self.destination_node
                    self.geometry = self.destination_geometry
                    self.visited_pois.append(self.destination_node)
                    
                    # Add resident to POI's visitors if it's a POI agent
                    visited_poi_type = None
                    visited_poi_agent = None
                    for poi in self.model.poi_agents:
                        if poi.node_id == self.destination_node and hasattr(poi, 'visitors'):
                            poi.visitors.add(self.unique_id)
                            visited_poi_type = poi.poi_type
                            visited_poi_agent = poi
                            # --- MEMORY MODULE: Record visit ---
                            self.memory['visited_pois'].append({
                                'step': getattr(self.model, 'step_count', None),
                                'poi_id': poi.unique_id,
                                'poi_type': poi.poi_type,
                                'category': getattr(poi, 'category', None),
                                'income': self.income
                            })
                            
                            # Track POI visit in output controller
                            if visited_poi_agent and hasattr(self.model, 'output_controller'):
                                poi_category = getattr(visited_poi_agent, 'category', 'other')
                                self.model.output_controller.track_poi_visit(poi_category)
                            
                            break
                    
                    # Start waiting at POI if it has waiting time
                    if visited_poi_agent and hasattr(visited_poi_agent, 'get_waiting_time'):
                        waiting_time = visited_poi_agent.get_waiting_time()
                        if waiting_time > 0:
                            self.waiting_at_poi = True
                            self.waiting_time_remaining = waiting_time
                            print(f"Resident {self.unique_id} waiting {waiting_time} minutes at {visited_poi_type}")
                            
                            # Track waiting time in output controller
                            if hasattr(self.model, 'output_controller'):
                                poi_category = getattr(visited_poi_agent, 'category', 'other')
                                self.model.output_controller.track_waiting_start(self.unique_id, poi_category, waiting_time)
                    
                    # Satisfy needs if we're using need-based movement and visited a POI
                    if self.movement_behavior == 'need-based' and visited_poi_type:
                        self._satisfy_needs_at_poi(visited_poi_type)
                    
                    # Reset travel attributes
                    self.destination_node = None
                    self.destination_geometry = None
                
                # Still traveling, don't take any other movement actions
                return
            
            # Handle waiting at POI
            if self.waiting_at_poi:
                self.waiting_time_remaining -= 1
                
                # Check if waiting is finished
                if self.waiting_time_remaining <= 0:
                    self.waiting_at_poi = False
                    print(f"Resident {self.unique_id} finished waiting at POI")
                
                # Still waiting, don't take any other movement actions
                return
            
            # Regular movement behavior - only if energy is not depleted and not traveling
            if self.energy > 0 and not self.traveling:
                # Choose movement target based on behavior setting
                target_poi_type = self.choose_movement_target()
                
                if target_poi_type == 'home':
                    # Force going home
                    self.go_home()
                elif target_poi_type:
                    # Move to the specified POI type
                    self.move_to_poi(target_poi_type)
                
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
    
    def generate_needs(self, method=None):
        """
        Generate needs for the resident based on the specified method.
        
        Args:
            method: The method to use for generating needs. If None, uses self.needs_selection.
                   Options: 'random', 'maslow', 'capability', 'llms'
                   
        Returns:
            Dictionary of generated needs
        """
        if method is None:
            method = self.needs_selection
        
        if method == 'random':
            return self._generate_needs_random()
        elif method == 'maslow':
            return self._generate_needs_maslow()
        elif method == 'capability':
            return self._generate_needs_capability()
        elif method == 'llms':
            return self._generate_needs_llms()
        else:
            # Default to random if unknown method
            return self._generate_needs_random()
    
    def _generate_needs_random(self):
        """
        Generate needs randomly.
        
        Returns:
            Dictionary of randomly generated needs
        """
        needs = {}
        for need_type in self.dynamic_needs.keys():
            # Generate random need level between 0 and 100
            needs[need_type] = random.randint(0, 100)
        
        return needs
    
    def _generate_needs_maslow(self):
        """
        Generate needs based on Maslow's hierarchy of needs.
        Prioritizes basic needs first, then higher-level needs.
        
        Returns:
            Dictionary of needs based on Maslow's hierarchy
        """
        needs = {}

        
        return needs
    
    def _generate_needs_capability(self):
        """
        Generate needs based on capability approach (Sen/Nussbaum).
        Focuses on what people are able to do and be.
        
        Returns:
            Dictionary of needs based on capability approach
        """
        needs = {}

        
        return needs
    
    def _generate_needs_llms(self):
        """
        Generate needs using LLM-inspired approach.
        This is a placeholder for future LLM integration.
        Currently uses a sophisticated rule-based system that mimics LLM reasoning.
        
        Returns:
            Dictionary of needs based on LLM-inspired reasoning
        """
        needs = {}
        

        
        return needs
            
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

    def go_home(self):
        """
        Move to the home node.
        
        Returns:
            Boolean indicating if the move home was successful
        """
        # If already at home or traveling, don't start a new journey
        if self.current_node == self.home_node or self.traveling:
            return False
        
        # Get home node coordinates for geometry
        node_coords = self.model.graph.nodes[self.home_node]
        home_geometry = None
        if 'x' in node_coords and 'y' in node_coords:
            from shapely.geometry import Point
            home_geometry = Point(node_coords['x'], node_coords['y'])
        
        # Start traveling home
        return self.start_travel(self.home_node, home_geometry)
