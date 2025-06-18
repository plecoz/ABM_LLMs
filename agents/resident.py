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

        # Employment probabilities - only for Macau
        if hasattr(self.model, 'city') and self.model.city == 'Macau, China':
            self.employment = {
                "No schooling / Pre-primary education": {
                    "employed": 0.204859253,
                    "unemployed or inactive": 0.795140747
                },
                "Primary education": {
                    "incomplete": {
                        "employed": 0.347652163,
                        "unemployed or inactive": 0.652347837
                    },
                    "complete": {
                        "employed": 0.487596097,
                        "unemployed or inactive": 0.512403903
                    }
                },
                "Secondary education": {
                    "Junior": {
                        "employed": 0.586663197,
                        "unemployed or inactive": 0.413336803
                    },
                    "Senior": {
                        "employed": 0.607928076,
                        "unemployed or inactive": 0.392071924
                    }
                },
                "Diploma programme": {
                    "employed": 0.747351695,
                    "unemployed or inactive": 0.252648305
                },
                "Tertiary education": {
                    "employed": 0.801285549,
                    "unemployed or inactive": 0.198714451
                },
                "Others": {
                    "employed": 0.196492271,
                    "unemployed or inactive": 0.803507729
                }
            }
        else:
            self.employment = None
        
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
        self.age_class = kwargs.get('age_class', None)  # Age class from parish demographics (e.g., "20-24")
        self.gender = kwargs.get('gender', 'male')  # Default gender
        self.income = kwargs.get('income', 50000)  # Default income
        self.education = kwargs.get('education', 'high_school')  # Default education level
        
        # New attributes
        # Using default values for employment status and household type
        self.employment_status = kwargs.get('employment_status', "employed")
        self.household_type = kwargs.get('household_type', "single")
        
        # Determine step size based on age for calculating travel times
        is_elderly = False
        if self.age_class:
            age_class_str = str(self.age_class).lower()
            if any(s in age_class_str for s in ['65+', '65-', '70+', '70-', '75+', '75-', '80+', '80-', '85+', '85-']):
                is_elderly = True
        
        self.step_size = 60.0 if is_elderly else 80.0  # meters per minute

        # Calculate home access time penalty based on distance from building to network
        self.access_distance = kwargs.get('access_distance', 0)
        if self.access_distance > 0.1:  # Only apply penalty for distances > 0.1 meters
            # Time (in steps/minutes) to walk from building to nearest street node
            # We use ceil to ensure any non-zero distance results in at least a 1-minute penalty
            self.home_access_time = math.ceil(self.access_distance / self.step_size)
            print(f"DEBUG: Resident {self.unique_id} has a home access time penalty of {self.home_access_time} minutes.")
        else:
            self.home_access_time = 0
        
        # Energy levels and mobility constraints
        # self.max_energy = 100
        # self.energy = self.max_energy
        # Age-based energy depletion rate: older agents lose energy faster
        # if self.age < 18:
        #     self.energy_depletion_rate = 2  # Children have moderate depletion
        # elif self.age < 35:
        #     self.energy_depletion_rate = 1  # Young adults have lowest depletion
        # elif self.age < 65:
        #     self.energy_depletion_rate = 2  # Middle-aged adults have moderate depletion
        # else:
        #     self.energy_depletion_rate = 3  # Elderly have highest depletion
        
        # Recharge counter when at home
        # self.home_recharge_counter = 0
        
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
        Calculate the travel time between two nodes based on age-class-specific step sizes.
        - Residents with age_class indicating 65+: 60-meter steps (1 minute at 3.6km/h)
        - Younger residents: 80-meter steps (1 minute at 4.8km/h)
        Always rounds up to ensure the agent doesn't move more than their step size per minute.
        
        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            
        Returns:
            Number of time steps needed for travel (each step is 1 minute)
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

        # Determine step size based on age_class
        # Check if age_class indicates elderly (65+ years)
        is_elderly = False
        if self.age_class:
            age_class_str = str(self.age_class).lower()
            # Check for age classes that indicate 65+ years
            if ('65+' in age_class_str or '65-' in age_class_str or 
                '70+' in age_class_str or '70-' in age_class_str or
                '75+' in age_class_str or '75-' in age_class_str or
                '80+' in age_class_str or '80-' in age_class_str or
                '85+' in age_class_str or '85-' in age_class_str):
                is_elderly = True
        
        if is_elderly:
            step_size = 60.0  # 60 meters per minute for elderly
        else:
            step_size = 80.0  # 80 meters per minute for younger residents
        
        # Calculate number of steps needed
        # Always round UP to ensure no step exceeds the agent's step size
        steps_needed = math.ceil(distance_meters / step_size)
        
        # Debug output for resident 0
        #if hasattr(self, 'unique_id') and self.unique_id == 0:
        #    print(f"Resident 0 (age_class {self.age_class}): Distance {distance_meters:.1f}m → {steps_needed} steps ({step_size}m each)")
        
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
        
        # Add access time penalty if starting from or going to home
        is_starting_from_home = self.current_node == self.home_node and not self.traveling
        is_going_home = target_node == self.home_node
        
        if (is_starting_from_home or is_going_home) and self.current_node != target_node:
            print(f"DEBUG: Resident {self.unique_id} trip to/from home. Base travel time: {travel_time} min. Adding access penalty: {self.home_access_time} min.")
            travel_time += self.home_access_time
        
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
            "recreation": ["park", "cinema", "theatre", "sports_centre", "museum", "library", "tourist_attraction", "casino"],
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
        elif self.movement_behavior == 'llms':
            return self._choose_llm_based_target()
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

    def _choose_llm_based_target(self):
        """
        Choose movement target using LLM-based decision making.
        
        Returns:
            POI type to move to, 'home' to go home, or None if no movement should occur
        """
        # Check if LLM components are available
        if not hasattr(self.model, 'llm_interaction_layer') or not self.model.llm_interaction_layer:
            self.logger.warning("LLM interaction layer not available, falling back to need-based movement")
            return self._choose_need_based_target()
        
        try:
            # Create agent state for LLM
            agent_state = self.model.llm_interaction_layer.create_agent_state_from_resident(self)
            
            # Create observation for LLM
            observation = self.model.llm_interaction_layer.create_observation_from_context(self, self.model)
            
            # Get episodic memories (recent POI visits)
            episodic_memories = self._get_episodic_memories()
            
            # Get LLM decision
            decision = self.model.llm_interaction_layer.get_agent_decision(
                agent_state=agent_state,
                observation=observation,
                episodic_memories=episodic_memories,
                agent_complexity="standard",
                latency_requirement="normal"
            )
            
            # Parse the LLM decision to extract POI type or action
            target = self._parse_llm_decision(decision)
            
            # Update emotional state based on decision confidence
            if hasattr(self, 'emotional_state') and hasattr(self.model, 'persona_memory_manager'):
                experience = {
                    'type': 'decision_making',
                    'outcome': 'positive' if decision.confidence > 0.7 else 'neutral',
                    'satisfaction': decision.confidence,
                    'details': f"Made decision: {decision.action}"
                }
                self.model.persona_memory_manager.update_agent_experience(str(self.unique_id), experience)
            
            return target
            
        except Exception as e:
            self.logger.error(f"Error in LLM-based target selection: {e}")
            # Fall back to need-based movement
            return self._choose_need_based_target()
    
    def _get_episodic_memories(self):
        """
        Get episodic memories for LLM decision making.
        
        Returns:
            List of EpisodicMemory objects
        """
        from simulation.llm_interaction_layer import EpisodicMemory
        
        memories = []
        
        # Convert recent POI visits to episodic memories
        if hasattr(self, 'memory') and 'visited_pois' in self.memory:
            recent_visits = self.memory['visited_pois'][-5:]  # Last 5 visits
            
            for visit in recent_visits:
                memory = EpisodicMemory(
                    timestamp=visit.get('step', 0),
                    location=str(visit.get('poi_id', 'unknown')),
                    action=f"visited_{visit.get('poi_type', 'unknown')}",
                    outcome="completed",
                    satisfaction_gained=0.7,  # Default satisfaction
                    other_agents_involved=[]
                )
                memories.append(memory)
        
        return memories
    
    def _parse_llm_decision(self, decision):
        """
        Parse LLM decision to extract movement target.
        
        Args:
            decision: LLMDecision object
            
        Returns:
            POI type to move to, 'home', or None
        """
        action = decision.action.lower()
        
        # Check if the action is to go home
        if 'home' in action or 'return' in action:
            return 'home'
        
        # Check if the action is to wait/stay
        if 'wait' in action or 'stay' in action or 'rest' in action:
            return None
        
        # Try to extract POI type from the action
        poi_types = ['restaurant', 'cafe', 'shop', 'hospital', 'school', 'park', 
                    'library', 'cinema', 'gym', 'pharmacy', 'bank', 'supermarket']
        
        for poi_type in poi_types:
            if poi_type in action:
                return poi_type
        
        # If no specific POI type found, try to infer from action keywords
        if 'eat' in action or 'food' in action or 'hungry' in action:
            return 'restaurant'
        elif 'shop' in action or 'buy' in action:
            return 'shop'
        elif 'health' in action or 'medical' in action:
            return 'hospital'
        elif 'exercise' in action or 'fitness' in action:
            return 'gym'
        elif 'relax' in action or 'nature' in action:
            return 'park'
        
        # Default: return None to stay put
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
                            
                            # Track waiting time in output controller
                            if hasattr(self.model, 'output_controller'):
                                poi_category = getattr(visited_poi_agent, 'category', 'other')
                                self.model.output_controller.track_waiting_start(self.unique_id, poi_category, waiting_time)
                    
                    # Satisfy needs if we're using need-based movement and visited a POI
                    if self.movement_behavior == 'need-based' and visited_poi_type:
                        self._satisfy_needs_at_poi(visited_poi_type)
                    
                    # Update emotional state if LLM behavior is enabled
                    if hasattr(self, 'emotional_state') and hasattr(self.model, 'persona_memory_manager'):
                        self._update_emotional_state_from_poi_visit(visited_poi_type, visited_poi_agent)
                    
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
            
            # === MOVEMENT DECISION MAKING ===
            # This is where all movement decisions are made based on the movement behavior
            if not self.traveling:
                target_poi_type = None
                
                if self.movement_behavior == 'random':
                    # Random movement - use existing simple logic
                    target_poi_type = self._make_random_movement_decision()
                    
                elif self.movement_behavior == 'need-based':
                    # Need-based movement - use existing logic but centralized here
                    target_poi_type = self._make_need_based_movement_decision()
                    
                elif self.movement_behavior == 'llms':
                    # LLM-based movement - placeholder for future sophisticated decision making
                    target_poi_type = self._make_llm_movement_decision()
                
                # Execute the movement decision
                if target_poi_type == 'home':
                    self.go_home()
                elif target_poi_type:
                    self.move_to_poi(target_poi_type)
                # If target_poi_type is None, resident stays put this step
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in resident step: {e}")

    def _make_random_movement_decision(self):
        """
        Make a random movement decision.
        Uses the existing random movement logic but with step-level decision making.
        
        Returns:
            POI type to move to, 'home' to go home, or None to stay put
        """
        # Simple probability check for random movement
        if random.random() > 0.3:  # 70% chance to stay put for random movement
            return None
        
        # Use existing random target selection
        target = self._choose_random_target()
        if target is None and self.current_node != self.home_node:
            # No suitable POI available and not at home - force going home
            return 'home'
        return target

    def _make_need_based_movement_decision(self):
        """
        Make a need-based movement decision.
        This is where we can later add sophisticated need hierarchy and POI attractiveness.
        
        Returns:
            POI type to move to, 'home' to go home, or None to stay put
        """
        # Update current needs
        self.current_needs = self.generate_needs()
        
        # Find the highest need
        highest_need_type, highest_need_value = self.find_highest_need()
        
        # Basic decision factors (can be expanded later)
        base_move_probability = 0.1  # Base 10% chance to move
        
        # Adjust based on time of day
        hour = self.model.hour_of_day
        time_factor = 1.0
        if 6 <= hour <= 22:  # Daytime hours
            time_factor = 1.5
        elif 22 < hour or hour < 6:  # Night hours
            time_factor = 0.2
        
        # Adjust based on age
        age_factor = 1.0
        if self.age >= 65:
            age_factor = 0.7
        elif self.age < 25:
            age_factor = 1.3
        
        # Adjust based on current location
        location_factor = 1.0
        if self.current_node == self.home_node:
            location_factor = 0.8  # Less likely to leave home
        else:
            location_factor = 1.2  # More likely to move when already out
        
        # Adjust based on need urgency (this is where hierarchy comes in)
        need_factor = 1.0
        if highest_need_value > 80:
            need_factor = 3.0  # Very urgent needs
        elif highest_need_value > 60:
            need_factor = 2.0  # Moderate needs
        elif highest_need_value > 40:
            need_factor = 1.2  # Mild needs
        elif highest_need_value < 30:
            need_factor = 0.3  # Low needs
        
        # Calculate final probability
        move_probability = base_move_probability * time_factor * age_factor * location_factor * need_factor
        move_probability = min(1.0, move_probability)  # Cap at 100%
        
        # Make the decision
        if random.random() > move_probability:
            return None  # Stay put
        
        # If we decide to move, find POI types that can satisfy the highest need
        available_poi_types = self.find_poi_for_need(highest_need_type)
        
        if available_poi_types:
            # TODO: This is where POI attractiveness/popularity could be considered
            # For now, choose randomly from available options
            return random.choice(available_poi_types)
        
        return None

    def _make_llm_movement_decision(self):
        """
        Make an LLM-based movement decision.
        This is a placeholder for future sophisticated LLM decision making.
        
        In the future, this could:
        - Evaluate specific POI options with their attractiveness
        - Consider complex need hierarchies
        - Factor in social context, weather, events, etc.
        - Make more human-like decisions
        
        Returns:
            POI type to move to, 'home' to go home, or None to stay put
        """
        # For now, fall back to existing LLM logic
        # This will be replaced with more sophisticated decision making later
        try:
            return self._choose_llm_based_target()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"LLM decision making failed, falling back to need-based: {e}")
            return self._make_need_based_movement_decision()

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
        Uses persona template and emotional state for more realistic need generation.
        
        Returns:
            Dictionary of needs based on LLM-inspired reasoning
        """
        needs = {}
        
        # Initialize with base needs
        for need_type in self.dynamic_needs.keys():
            needs[need_type] = 0
        
        # Check if persona components are available
        if not hasattr(self, 'persona_template') or not hasattr(self, 'emotional_state'):
            # Fall back to random generation if persona not available
            return self._generate_needs_random()
        
        try:
            # Get persona-specific needs based on persona type
            persona_needs = self._get_persona_based_needs()
            
            # Adjust needs based on emotional state
            emotional_adjustments = self._get_emotional_need_adjustments()
            
            # Combine persona needs with emotional adjustments
            for need_type in needs.keys():
                base_need = persona_needs.get(need_type, 30)  # Default moderate need
                emotional_modifier = emotional_adjustments.get(need_type, 1.0)
                
                # Apply emotional modifier
                adjusted_need = base_need * emotional_modifier
                
                # Add some randomness for variability
                random_factor = random.uniform(0.8, 1.2)
                final_need = adjusted_need * random_factor
                
                # Clamp to valid range
                needs[need_type] = max(0, min(100, int(final_need)))
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in LLM needs generation: {e}")
            # Fall back to random generation
            return self._generate_needs_random()
        
        return needs
    
    def _get_persona_based_needs(self):
        """
        Get base needs based on persona type.
        
        Returns:
            Dictionary of base needs for the persona
        """
        from agents.persona_memory_modules import PersonaType
        
        persona_type = getattr(self, 'persona_type', None)
        
        # Default needs
        base_needs = {
            "hunger": 40,
            "social": 35,
            "recreation": 30,
            "shopping": 25,
            "healthcare": 20,
            "education": 15
        }
        
        if persona_type == PersonaType.ELDERLY_RESIDENT:
            base_needs.update({
                "healthcare": 60,  # Higher healthcare needs
                "social": 45,      # Higher social needs
                "recreation": 25,  # Lower recreation needs
                "shopping": 35,    # Moderate shopping needs
                "hunger": 45,      # Regular hunger needs
                "education": 10    # Lower education needs
            })
        elif persona_type == PersonaType.WORKING_PARENT:
            base_needs.update({
                "shopping": 50,    # Higher shopping needs (family)
                "healthcare": 35,  # Moderate healthcare needs
                "social": 30,      # Lower social needs (busy)
                "recreation": 20,  # Lower recreation needs
                "hunger": 50,      # Higher hunger needs
                "education": 25    # Moderate education needs (children)
            })
        elif persona_type == PersonaType.YOUNG_PROFESSIONAL:
            base_needs.update({
                "recreation": 45,  # Higher recreation needs
                "social": 40,      # Higher social needs
                "shopping": 35,    # Moderate shopping needs
                "healthcare": 25,  # Lower healthcare needs
                "hunger": 40,      # Regular hunger needs
                "education": 30    # Moderate education needs
            })
        elif persona_type == PersonaType.STUDENT:
            base_needs.update({
                "education": 55,   # Higher education needs
                "social": 50,      # Higher social needs
                "recreation": 40,  # Higher recreation needs
                "shopping": 20,    # Lower shopping needs (budget)
                "healthcare": 20,  # Lower healthcare needs
                "hunger": 45       # Regular hunger needs
            })
        elif persona_type == PersonaType.CHRONIC_PATIENT:
            base_needs.update({
                "healthcare": 70,  # Very high healthcare needs
                "social": 40,      # Moderate social needs (support)
                "recreation": 25,  # Lower recreation needs
                "shopping": 30,    # Moderate shopping needs
                "hunger": 40,      # Regular hunger needs
                "education": 20    # Lower education needs
            })
        
        return base_needs
    
    def _get_emotional_need_adjustments(self):
        """
        Get need adjustments based on current emotional state.
        
        Returns:
            Dictionary of multipliers for each need type
        """
        adjustments = {
            "hunger": 1.0,
            "social": 1.0,
            "recreation": 1.0,
            "shopping": 1.0,
            "healthcare": 1.0,
            "education": 1.0
        }
        
        if not hasattr(self, 'emotional_state'):
            return adjustments
        
        try:
            # Get dominant emotion and stress level
            emotions = self.emotional_state.current_emotions
            stress_level = self.emotional_state.stress_level
            
            # Find dominant emotion
            if emotions:
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                
                # Adjust needs based on dominant emotion
                from agents.persona_memory_modules import EmotionalState
                
                if dominant_emotion == EmotionalState.STRESSED:
                    adjustments["recreation"] *= 1.3  # More recreation when stressed
                    adjustments["social"] *= 0.8      # Less social when stressed
                    adjustments["healthcare"] *= 1.2  # More healthcare when stressed
                elif dominant_emotion == EmotionalState.ANXIOUS:
                    adjustments["healthcare"] *= 1.4  # More healthcare when anxious
                    adjustments["social"] *= 1.2      # More social support when anxious
                    adjustments["recreation"] *= 0.9  # Less recreation when anxious
                elif dominant_emotion == EmotionalState.FRUSTRATED:
                    adjustments["recreation"] *= 1.2  # More recreation when frustrated
                    adjustments["shopping"] *= 1.1    # Slight increase in shopping (retail therapy)
                elif dominant_emotion == EmotionalState.SATISFIED:
                    adjustments["recreation"] *= 0.9  # Less urgent recreation when satisfied
                    adjustments["social"] *= 1.1      # More social when satisfied
                elif dominant_emotion == EmotionalState.WORRIED:
                    adjustments["healthcare"] *= 1.3  # More healthcare when worried
                    adjustments["social"] *= 1.2      # More social support when worried
            
            # Adjust based on stress level
            if stress_level > 0.7:  # High stress
                adjustments["recreation"] *= 1.2
                adjustments["healthcare"] *= 1.1
            elif stress_level < 0.3:  # Low stress
                adjustments["recreation"] *= 0.9
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in emotional need adjustments: {e}")
        
        return adjustments
            
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
                "age_class": self.age_class,
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

    def _update_emotional_state_from_poi_visit(self, poi_type, poi_agent):
        """
        Update emotional state based on a POI visit.
        
        Args:
            poi_type: The type of POI visited
            poi_agent: The POI agent visited
        """
        try:
            # Determine satisfaction based on POI type and waiting time
            base_satisfaction = 0.7  # Default satisfaction
            
            # Adjust satisfaction based on POI type
            poi_satisfaction_map = {
                'restaurant': 0.8,
                'cafe': 0.7,
                'park': 0.8,
                'hospital': 0.6,  # Lower satisfaction for healthcare visits
                'shop': 0.7,
                'supermarket': 0.6,
                'library': 0.8,
                'cinema': 0.9,
                'gym': 0.8,
                'school': 0.7
            }
            
            base_satisfaction = poi_satisfaction_map.get(poi_type, 0.7)
            
            # Adjust satisfaction based on waiting time
            if poi_agent and hasattr(poi_agent, 'get_waiting_time'):
                waiting_time = poi_agent.get_waiting_time()
                if waiting_time > 30:  # Long wait reduces satisfaction
                    base_satisfaction *= 0.7
                elif waiting_time > 15:  # Moderate wait slightly reduces satisfaction
                    base_satisfaction *= 0.9
            
            # Create experience for emotional state update
            experience = {
                'type': 'healthcare_visit' if poi_type in ['hospital', 'clinic', 'pharmacy'] else 'service_interaction',
                'outcome': 'positive' if base_satisfaction > 0.6 else 'neutral',
                'satisfaction': base_satisfaction,
                'details': f"Visited {poi_type}"
            }
            
            # Update emotional state through persona memory manager
            self.model.persona_memory_manager.update_agent_experience(str(self.unique_id), experience)
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error updating emotional state from POI visit: {e}")
