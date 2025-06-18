from mesa.agent import Agent
from agents.base_agent import BaseAgent
import random
import networkx as nx
import logging
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

class ActionType(Enum):
    """Types of actions residents can perform"""
    # Basic actions (always available)
    WAITING = "waiting"
    TRAVELING = "traveling"
    SLEEPING = "sleeping"
    
    # POI-specific actions
    EATING = "eating"
    SHOPPING = "shopping"
    WORKING = "working"
    STUDYING = "studying"
    EXERCISING = "exercising"
    SOCIALIZING = "socializing"
    HEALTHCARE = "healthcare"
    ENTERTAINMENT = "entertainment"
    WORSHIP = "worship"
    BANKING = "banking"
    
    # Granular actions (for detailed simulations)
    TALKING = "talking"
    READING = "reading"
    WATCHING_MOVIE = "watching_movie"
    DOING_BUSINESS = "doing_business"
    BROWSING = "browsing"
    CONSULTING = "consulting"
    PRAYING = "praying"
    PLAYING = "playing"

class ActionGranularity(Enum):
    """Granularity levels for action simulation"""
    SIMPLE = "simple"      # Just waiting times
    BASIC = "basic"        # Basic action types (eating, shopping, etc.)
    DETAILED = "detailed"  # Granular actions (talking, browsing, etc.)

@dataclass
class Action:
    """Represents an action being performed by a resident"""
    action_type: ActionType
    duration: int  # Duration in time steps (minutes)
    poi_type: Optional[str] = None
    poi_id: Optional[int] = None
    description: Optional[str] = None
    needs_satisfied: Optional[Dict[str, int]] = None
    social_interaction: bool = False
    other_agents: Optional[List[int]] = None
    context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.needs_satisfied is None:
            self.needs_satisfied = {}
        if self.other_agents is None:
            self.other_agents = []
        if self.context is None:
            self.context = {}

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
        
        # Core location and mobility attributes (keep separate for frequent access)
        self.home_node = home_node
        self.current_node = home_node
        self.accessible_nodes = accessible_nodes
        self.visited_pois = []
        self.mobility_mode = "walk"
        self.last_visited_node = None
        
        # Household and occupation (keep separate as requested)
        self.household_members = kwargs.get('household_members', [])
        self.household_type = kwargs.get('household_type', "single")
        self.employment_status = kwargs.get('employment_status', "employed")
        
        # Consolidated attributes dictionary
        self.attributes = {
            # Demographics
            'age': kwargs.get('age', 30),
            'age_class': kwargs.get('age_class', None),
            'gender': kwargs.get('gender', 'male'),
            'income': kwargs.get('income', 50000),
            'education': kwargs.get('education', 'high_school'),
            
            # Location and social
            'parish': kwargs.get('parish', None),
            'family_id': kwargs.get('family_id', None),
            'social_network': kwargs.get('social_network', []),
            
            # Behavior and preferences
            'needs_selection': kwargs.get('needs_selection', 'random'),
            'movement_behavior': kwargs.get('movement_behavior', 'need-based'),
            'daily_schedule': kwargs.get('daily_schedule', {}),
            'personality_traits': kwargs.get('personality_traits', {}),
            'activity_preferences': kwargs.get('activity_preferences', {}),
            
            # Physical attributes
            'access_distance': kwargs.get('access_distance', 0),
        }
        
        # Convenience properties for frequently accessed attributes
        self.age = self.attributes['age']
        self.parish = self.attributes['parish']
        self.needs_selection = self.attributes['needs_selection']
        self.movement_behavior = self.attributes['movement_behavior']
        self.social_network = self.attributes['social_network']

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
        
        # Determine step size based on age for calculating travel times
        is_elderly = False
        if self.attributes['age_class']:
            age_class_str = str(self.attributes['age_class']).lower()
            if any(s in age_class_str for s in ['65+', '65-', '70+', '70-', '75+', '75-', '80+', '80-', '85+', '85-']):
                is_elderly = True
        
        self.step_size = 60.0 if is_elderly else 80.0  # meters per minute

        # Calculate home access time penalty based on distance from building to network
        if self.attributes['access_distance'] > 0.1:  # Only apply penalty for distances > 0.1 meters
            # Time (in steps/minutes) to walk from building to nearest street node
            # We use ceil to ensure any non-zero distance results in at least a 1-minute penalty
            self.home_access_time = math.ceil(self.attributes['access_distance'] / self.step_size)
            print(f"DEBUG: Resident {self.unique_id} has a home access time penalty of {self.home_access_time} minutes.")
        else:
            self.home_access_time = 0
        
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
        
        # Action system (replaces simple waiting)
        self.performing_action = False
        self.current_action = None  # Current action being performed
        self.action_time_remaining = 0
        self.action_history = []  # Track completed actions
        
        # Path selection for LLM agents
        self.selected_travel_path = None  # Store the path selected by LLM
        self.path_selection_history = []  # Track path choices for learning
        
        # Enhanced memory module
        self.memory = {
            'income': self.attributes['income'],
            'visited_pois': [],  # List of dicts: {step, poi_id, poi_type, category, income}
            'interactions': [],  # List of interaction records
            'historical_needs': [],  # List of needs over time: {step, needs_dict}
            'completed_actions': [],  # List of completed actions with outcomes
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
        # For LLM-enabled agents, use path selection instead of simple shortest path
        if self.movement_behavior == 'llms' and hasattr(self.model, 'llm_interaction_layer'):
            return self._calculate_travel_time_with_path_selection(from_node, to_node)
        
        # Standard shortest path calculation for non-LLM agents
        return self._calculate_standard_travel_time(from_node, to_node)

    def _calculate_standard_travel_time(self, from_node, to_node):
        """
        Calculate travel time using standard shortest path (for non-LLM agents).
        
        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            
        Returns:
            Number of time steps needed for travel
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
        if self.attributes['age_class']:
            age_class_str = str(self.attributes['age_class']).lower()
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
        
        # Ensure at least 1 time step
        return max(1, steps_needed)

    def _calculate_travel_time_with_path_selection(self, from_node, to_node):
        """
        Calculate travel time using LLM-based path selection from multiple alternatives.
        
        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            
        Returns:
            Number of time steps needed for travel using selected path
        """
        try:
            # Get multiple path options
            path_options = self._get_multiple_path_options(from_node, to_node)
            
            if not path_options:
                self.logger.warning(f"No path options found from {from_node} to {to_node}")
                return None
            
            # If only one path available, use it directly
            if len(path_options) == 1:
                selected_path = path_options[0]
            else:
                # Use LLM to score and select the best path
                selected_path = self._select_path_with_llm(path_options, from_node, to_node)
            
            # Store the selected path for actual travel
            self.selected_travel_path = selected_path
            
            # Calculate travel time based on selected path
            return self._calculate_time_for_path(selected_path)
            
        except Exception as e:
            self.logger.error(f"Error in LLM path selection: {e}")
            # Fall back to standard calculation
            return self._calculate_standard_travel_time(from_node, to_node)

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
            POI type to move to, 'home' to go home, or None to stay put
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
                                'income': self.attributes['income']
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
                            # Determine action based on POI type and simulation settings
                            action = self._select_action_at_poi(visited_poi_agent, waiting_time)
                            self._start_action(action)
                            
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
            
            # Handle ongoing actions at POIs
            if self.performing_action:
                self.action_time_remaining -= 1
                
                # Check if action is finished
                if self.action_time_remaining <= 0:
                    self._complete_current_action()
                
                # Still performing action, don't take any other movement actions
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
            
            # Record needs snapshot periodically (every 15 minutes)
            if self.model.step_count % 15 == 0:
                self.record_needs_snapshot()
            
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
        self.attributes['activity_preferences'] = preferences
    
    def add_to_social_network(self, agent_id):
        """
        Add an agent to this agent's social network.
        
        Args:
            agent_id: ID of the agent to add
        """
        if agent_id != self.unique_id and agent_id not in self.social_network:
            self.social_network.append(agent_id)
            self.attributes['social_network'] = self.social_network  # Keep attributes dict in sync
    
    def remove_from_social_network(self, agent_id):
        """
        Remove an agent from this agent's social network.
        
        Args:
            agent_id: ID of the agent to remove
        """
        if agent_id in self.social_network:
            self.social_network.remove(agent_id)
            self.attributes['social_network'] = self.social_network  # Keep attributes dict in sync
    
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
                "age_class": self.attributes['age_class'],
                "gender": self.attributes['gender'],
                "income": self.attributes['income'],
                "education": self.attributes['education'],
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

    def record_interaction(self, other_agent_id, interaction_type, message=None, context=None):
        """
        Record an interaction with another agent.
        
        Args:
            other_agent_id: ID of the other agent involved
            interaction_type: Type of interaction ('message', 'meeting', 'service', etc.)
            message: The message content (if applicable)
            context: Additional context information
        """
        interaction_record = {
            'step': getattr(self.model, 'step_count', 0),
            'timestamp': self.model.get_current_time() if hasattr(self.model, 'get_current_time') else None,
            'other_agent_id': other_agent_id,
            'interaction_type': interaction_type,
            'message': message,
            'context': context,
            'my_location': self.current_node,
            'my_parish': self.parish
        }
        
        self.memory['interactions'].append(interaction_record)
        
        # Keep only last 100 interactions to prevent memory bloat
        if len(self.memory['interactions']) > 100:
            self.memory['interactions'] = self.memory['interactions'][-100:]

    def record_needs_snapshot(self):
        """
        Record current needs state for historical tracking.
        Called periodically to track how needs evolve over time.
        """
        needs_snapshot = {
            'step': getattr(self.model, 'step_count', 0),
            'timestamp': self.model.get_current_time() if hasattr(self.model, 'get_current_time') else None,
            'needs': self.current_needs.copy(),
            'location': self.current_node,
            'at_home': self.current_node == self.home_node,
            'traveling': self.traveling
        }
        
        self.memory['historical_needs'].append(needs_snapshot)
        
        # Keep only last 200 snapshots (roughly 3+ hours if recorded every minute)
        if len(self.memory['historical_needs']) > 200:
            self.memory['historical_needs'] = self.memory['historical_needs'][-200:]

    def get_recent_interactions(self, interaction_type=None, limit=10):
        """
        Get recent interactions, optionally filtered by type.
        
        Args:
            interaction_type: Filter by interaction type (optional)
            limit: Maximum number of interactions to return
            
        Returns:
            List of recent interaction records
        """
        interactions = self.memory['interactions']
        
        if interaction_type:
            interactions = [i for i in interactions if i['interaction_type'] == interaction_type]
        
        return interactions[-limit:] if interactions else []

    def get_needs_history(self, steps_back=60):
        """
        Get historical needs data for analysis.
        
        Args:
            steps_back: How many steps back to retrieve (default: 60 = 1 hour)
            
        Returns:
            List of needs snapshots
        """
        current_step = getattr(self.model, 'step_count', 0)
        cutoff_step = current_step - steps_back
        
        return [
            snapshot for snapshot in self.memory['historical_needs']
            if snapshot['step'] >= cutoff_step
        ]

    def get_attributes_dict(self):
        """
        Get a copy of the attributes dictionary for external use.
        Useful for LLM integration and serialization.
        
        Returns:
            Dictionary copy of agent attributes
        """
        return self.attributes.copy()

    def update_attribute(self, key, value):
        """
        Update an attribute and sync convenience properties if needed.
        
        Args:
            key: Attribute key to update
            value: New value
        """
        self.attributes[key] = value
        
        # Update convenience properties that might be affected
        if key == 'age':
            self.age = value
        elif key == 'parish':
            self.parish = value
        elif key == 'needs_selection':
            self.needs_selection = value
        elif key == 'movement_behavior':
            self.movement_behavior = value
        elif key == 'social_network':
            self.social_network = value

    def _select_action_at_poi(self, poi_agent, waiting_time):
        """
        Select an action to perform at a POI based on simulation granularity and LLM settings.
        
        Args:
            poi_agent: The POI agent visited
            waiting_time: The base waiting time at the POI
            
        Returns:
            Action object representing the selected action
        """
        # Get simulation granularity setting (default to BASIC)
        granularity = getattr(self.model, 'action_granularity', ActionGranularity.BASIC)
        
        # Simple granularity - just waiting
        if granularity == ActionGranularity.SIMPLE:
            return Action(
                action_type=ActionType.WAITING,
                duration=waiting_time,
                poi_type=poi_agent.poi_type,
                poi_id=poi_agent.unique_id,
                description=f"Waiting at {poi_agent.poi_type}"
            )
        
        # LLM-based action selection (if enabled)
        if self.movement_behavior == 'llms' and hasattr(self.model, 'llm_interaction_layer'):
            return self._select_llm_action(poi_agent, waiting_time)
        
        # Rule-based action selection for BASIC and DETAILED granularity
        return self._select_rule_based_action(poi_agent, waiting_time, granularity)

    def _select_rule_based_action(self, poi_agent, waiting_time, granularity):
        """
        Select action using rule-based logic based on POI type and granularity.
        
        Args:
            poi_agent: The POI agent visited
            waiting_time: The base waiting time
            granularity: The simulation granularity level
            
        Returns:
            Action object
        """
        poi_type = poi_agent.poi_type.lower()
        
        # Map POI types to possible actions
        poi_action_mapping = {
            'restaurant': {
                ActionGranularity.BASIC: [ActionType.EATING, ActionType.SOCIALIZING],
                ActionGranularity.DETAILED: [ActionType.EATING, ActionType.TALKING, ActionType.SOCIALIZING]
            },
            'cafe': {
                ActionGranularity.BASIC: [ActionType.EATING, ActionType.SOCIALIZING],
                ActionGranularity.DETAILED: [ActionType.EATING, ActionType.TALKING, ActionType.READING]
            },
            'shop': {
                ActionGranularity.BASIC: [ActionType.SHOPPING],
                ActionGranularity.DETAILED: [ActionType.SHOPPING, ActionType.BROWSING, ActionType.DOING_BUSINESS]
            },
            'supermarket': {
                ActionGranularity.BASIC: [ActionType.SHOPPING],
                ActionGranularity.DETAILED: [ActionType.SHOPPING, ActionType.BROWSING]
            },
            'hospital': {
                ActionGranularity.BASIC: [ActionType.HEALTHCARE],
                ActionGranularity.DETAILED: [ActionType.HEALTHCARE, ActionType.CONSULTING, ActionType.WAITING]
            },
            'clinic': {
                ActionGranularity.BASIC: [ActionType.HEALTHCARE],
                ActionGranularity.DETAILED: [ActionType.HEALTHCARE, ActionType.CONSULTING]
            },
            'school': {
                ActionGranularity.BASIC: [ActionType.STUDYING],
                ActionGranularity.DETAILED: [ActionType.STUDYING, ActionType.READING, ActionType.SOCIALIZING]
            },
            'library': {
                ActionGranularity.BASIC: [ActionType.STUDYING],
                ActionGranularity.DETAILED: [ActionType.READING, ActionType.STUDYING]
            },
            'cinema': {
                ActionGranularity.BASIC: [ActionType.ENTERTAINMENT],
                ActionGranularity.DETAILED: [ActionType.WATCHING_MOVIE, ActionType.SOCIALIZING]
            },
            'gym': {
                ActionGranularity.BASIC: [ActionType.EXERCISING],
                ActionGranularity.DETAILED: [ActionType.EXERCISING, ActionType.SOCIALIZING]
            },
            'park': {
                ActionGranularity.BASIC: [ActionType.ENTERTAINMENT, ActionType.EXERCISING],
                ActionGranularity.DETAILED: [ActionType.PLAYING, ActionType.EXERCISING, ActionType.SOCIALIZING]
            },
            'bank': {
                ActionGranularity.BASIC: [ActionType.BANKING],
                ActionGranularity.DETAILED: [ActionType.BANKING, ActionType.DOING_BUSINESS, ActionType.WAITING]
            },
            'place_of_worship': {
                ActionGranularity.BASIC: [ActionType.WORSHIP],
                ActionGranularity.DETAILED: [ActionType.WORSHIP, ActionType.PRAYING, ActionType.SOCIALIZING]
            }
        }
        
        # Get possible actions for this POI type and granularity
        possible_actions = poi_action_mapping.get(poi_type, {}).get(granularity, [ActionType.WAITING])
        
        # Select action (could be random or based on needs/preferences)
        selected_action_type = random.choice(possible_actions)
        
        # Determine needs satisfied based on action type
        needs_satisfied = self._get_needs_satisfied_by_action(selected_action_type, poi_type)
        
        # Create action with context
        return Action(
            action_type=selected_action_type,
            duration=waiting_time,
            poi_type=poi_type,
            poi_id=poi_agent.unique_id,
            description=f"{selected_action_type.value.replace('_', ' ').title()} at {poi_type}",
            needs_satisfied=needs_satisfied,
            social_interaction=selected_action_type in [ActionType.SOCIALIZING, ActionType.TALKING],
            context={'granularity': granularity.value}
        )

    def _select_llm_action(self, poi_agent, waiting_time):
        """
        Select action using LLM-based decision making.
        
        Args:
            poi_agent: The POI agent visited
            waiting_time: The base waiting time
            
        Returns:
            Action object selected by LLM
        """
        try:
            # This would integrate with your LLM system to make sophisticated action choices
            # For now, return a placeholder that falls back to rule-based selection
            granularity = getattr(self.model, 'action_granularity', ActionGranularity.DETAILED)
            
            # TODO: Implement actual LLM integration here
            # The LLM could consider:
            # - Current needs and emotional state
            # - Social context (other agents present)
            # - Time of day and personal schedule
            # - Past experiences at similar POIs
            # - Personal preferences and personality
            
            return self._select_rule_based_action(poi_agent, waiting_time, granularity)
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"LLM action selection failed, falling back to rule-based: {e}")
            return self._select_rule_based_action(poi_agent, waiting_time, ActionGranularity.BASIC)

    def _get_needs_satisfied_by_action(self, action_type, poi_type):
        """
        Determine which needs are satisfied by performing a specific action.
        
        Args:
            action_type: The type of action being performed
            poi_type: The type of POI where the action occurs
            
        Returns:
            Dictionary of needs satisfied and their amounts
        """
        needs_map = {
            ActionType.EATING: {'hunger': 40},
            ActionType.SHOPPING: {'shopping': 35},
            ActionType.SOCIALIZING: {'social': 30},
            ActionType.TALKING: {'social': 25},
            ActionType.HEALTHCARE: {'healthcare': 50},
            ActionType.CONSULTING: {'healthcare': 45},
            ActionType.STUDYING: {'education': 40},
            ActionType.READING: {'education': 30},
            ActionType.EXERCISING: {'recreation': 35},
            ActionType.ENTERTAINMENT: {'recreation': 40},
            ActionType.WATCHING_MOVIE: {'recreation': 45},
            ActionType.WORSHIP: {'social': 20, 'recreation': 15},
            ActionType.PRAYING: {'social': 15, 'recreation': 20},
            ActionType.BANKING: {'shopping': 25},  # Administrative needs
            ActionType.DOING_BUSINESS: {'shopping': 30},
            ActionType.BROWSING: {'shopping': 15, 'recreation': 10},
            ActionType.PLAYING: {'recreation': 35, 'social': 20},
            ActionType.WAITING: {}  # No needs satisfied by waiting
        }
        
        return needs_map.get(action_type, {})

    def _start_action(self, action):
        """
        Start performing an action.
        
        Args:
            action: The Action object to perform
        """
        self.performing_action = True
        self.current_action = action
        self.action_time_remaining = action.duration
        
        # Record action start
        if hasattr(self, 'logger'):
            self.logger.debug(f"Resident {self.unique_id} started {action.description} for {action.duration} minutes")
        
        # Record interaction if it's a social action
        if action.social_interaction:
            self.record_interaction(
                other_agent_id=action.poi_id,
                interaction_type='social_activity',
                context={'action': action.action_type.value, 'poi_type': action.poi_type}
            )

    def _complete_current_action(self):
        """
        Complete the current action being performed.
        """
        if not self.current_action:
            self.performing_action = False
            return
        
        action = self.current_action
        
        # Satisfy needs based on the action performed
        for need_type, satisfaction_amount in action.needs_satisfied.items():
            if need_type in self.current_needs:
                self.current_needs[need_type] = max(0, self.current_needs[need_type] - satisfaction_amount)
        
        # Record completed action in memory
        action_record = {
            'step': getattr(self.model, 'step_count', 0),
            'timestamp': self.model.get_current_time() if hasattr(self.model, 'get_current_time') else None,
            'action_type': action.action_type.value,
            'duration': action.duration,
            'poi_type': action.poi_type,
            'poi_id': action.poi_id,
            'description': action.description,
            'needs_satisfied': action.needs_satisfied,
            'social_interaction': action.social_interaction,
            'context': action.context
        }
        
        self.memory['completed_actions'].append(action_record)
        self.action_history.append(action_record)
        
        # Keep only last 50 completed actions to prevent memory bloat
        if len(self.memory['completed_actions']) > 50:
            self.memory['completed_actions'] = self.memory['completed_actions'][-50:]
        
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-50:]
        
        # Log completion
        if hasattr(self, 'logger'):
            self.logger.debug(f"Resident {self.unique_id} completed {action.description}")
        
        # Reset action state
        self.performing_action = False
        self.current_action = None
        self.action_time_remaining = 0

    def get_recent_actions(self, action_type=None, limit=10):
        """
        Get recent completed actions, optionally filtered by type.
        
        Args:
            action_type: Filter by action type (optional)
            limit: Maximum number of actions to return
            
        Returns:
            List of recent action records
        """
        actions = self.memory['completed_actions']
        
        if action_type:
            if isinstance(action_type, ActionType):
                action_type = action_type.value
            actions = [a for a in actions if a['action_type'] == action_type]
        
        return actions[-limit:] if actions else []

    def _get_multiple_path_options(self, from_node, to_node, max_paths=4):
        """
        Generate 4 shortest path options between two nodes.
        
        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            max_paths: Maximum number of paths to generate (default: 4)
            
        Returns:
            List of path dictionaries with OSM metadata for LLM scoring
        """
        try:
            # Generate k-shortest paths using NetworkX
            paths = self._get_k_shortest_paths(from_node, to_node, max_paths)
            
            # Extract OSM metadata for each path
            path_options = []
            for i, path_nodes in enumerate(paths):
                if path_nodes:  # Ensure path exists
                    path_data = self._extract_path_metadata(path_nodes, i + 1)
                    if path_data:
                        path_options.append(path_data)
            
            return path_options
            
        except Exception as e:
            self.logger.error(f"Error generating path options: {e}")
            return []

    def _get_k_shortest_paths(self, from_node, to_node, k):
        """
        Get k shortest paths using simple approach.
        
        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            k: Number of paths to find
            
        Returns:
            List of path node lists
        """
        try:
            import itertools
            
            paths = []
            graph = self.model.graph.copy()
            
            # Get the shortest path first
            try:
                shortest_path = nx.shortest_path(graph, from_node, to_node, weight='length')
                paths.append(shortest_path)
            except nx.NetworkXNoPath:
                return []
            
            # Try to find alternative paths by temporarily removing edges
            for attempt in range(k - 1):
                if len(paths) >= k:
                    break
                
                # Create a copy of the graph and remove some edges from previous paths
                temp_graph = graph.copy()
                
                # Remove some edges from existing paths to force alternatives
                for existing_path in paths:
                    if len(existing_path) > 2:  # Only if path has enough edges
                        # Remove middle edges to force different routes
                        edges_to_remove = []
                        for i in range(1, min(3, len(existing_path) - 1)):  # Remove 1-2 middle edges
                            if i < len(existing_path) - 1:
                                edges_to_remove.append((existing_path[i], existing_path[i + 1]))
                        
                        for edge in edges_to_remove:
                            if temp_graph.has_edge(edge[0], edge[1]):
                                temp_graph.remove_edge(edge[0], edge[1])
                
                # Try to find a path in the modified graph
                try:
                    alt_path = nx.shortest_path(temp_graph, from_node, to_node, weight='length')
                    # Check if this path is sufficiently different
                    if not any(self._paths_are_same(alt_path, existing) for existing in paths):
                        paths.append(alt_path)
                except nx.NetworkXNoPath:
                    continue
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error in k-shortest paths: {e}")
            return []

    def _extract_path_metadata(self, path_nodes, path_number):
        """
        Extract OSM metadata from a path for LLM scoring.
        
        Args:
            path_nodes: List of node IDs in the path
            path_number: Path identifier number
            
        Returns:
            Dictionary with path metadata including green area coverage
        """
        try:
            graph = self.model.graph
            
            # Calculate basic metrics
            path_length = 0
            road_types = []
            surface_types = []
            max_speeds = []
            
            # Analyze each edge in the path
            for i in range(len(path_nodes) - 1):
                node1, node2 = path_nodes[i], path_nodes[i + 1]
                
                # Get edge data
                edge_data = graph.get_edge_data(node1, node2, {})
                
                # Extract length
                edge_length = edge_data.get('length', 0)
                path_length += edge_length
                
                # Extract road type (highway tag)
                highway_type = edge_data.get('highway', 'unclassified')
                road_types.append(highway_type)
                
                # Extract surface type if available
                surface = edge_data.get('surface', 'unknown')
                surface_types.append(surface)
                
                # Extract max speed if available
                max_speed = edge_data.get('maxspeed', 'unknown')
                max_speeds.append(max_speed)
            
            # Calculate travel time
            travel_time_minutes = self._calculate_time_for_path_nodes(path_nodes)
            
            # Determine dominant road types
            road_type_counts = {}
            for road_type in road_types:
                road_type_counts[road_type] = road_type_counts.get(road_type, 0) + 1
            
            dominant_road_type = max(road_type_counts, key=road_type_counts.get) if road_type_counts else 'unknown'
            
            # Calculate green area coverage percentage
            green_area_percentage = self._calculate_green_area_coverage(path_nodes, path_length)
            
            # Create simplified metadata for LLM
            return {
                'path_id': path_number,
                'path_nodes': path_nodes,
                'distance_meters': round(path_length, 0),
                'travel_time_minutes': round(travel_time_minutes, 1),
                'dominant_road_type': dominant_road_type,
                'road_types': list(set(road_types)),
                'surface_types': list(set([s for s in surface_types if s != 'unknown'])),
                'has_speed_limits': any(speed != 'unknown' for speed in max_speeds),
                'total_segments': len(path_nodes) - 1,
                'green_area_percentage': round(green_area_percentage, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting path metadata: {e}")
            return None

    def _calculate_green_area_coverage(self, path_nodes, total_path_length):
        """
        Calculate the percentage of the path that goes through green areas.
        
        Args:
            path_nodes: List of node IDs in the path
            total_path_length: Total length of the path in meters
            
        Returns:
            Percentage of path in green areas (0-100)
        """
        try:
            # Check if green area data is available
            if not hasattr(self.model, 'forests') or self.model.forests is None or self.model.forests.empty:
                return 0.0
            
            from shapely.geometry import LineString, Point
            import geopandas as gpd
            
            # Create path geometry from nodes
            path_coordinates = []
            for node_id in path_nodes:
                node_data = self.model.graph.nodes[node_id]
                path_coordinates.append((node_data['x'], node_data['y']))
            
            # Create LineString from path coordinates
            path_line = LineString(path_coordinates)
            
            # Calculate intersection with green areas
            green_length = 0.0
            
            for _, green_area in self.model.forests.iterrows():
                try:
                    # Get intersection of path with green area
                    intersection = path_line.intersection(green_area.geometry)
                    
                    # Add length if there's an intersection
                    if not intersection.is_empty:
                        if hasattr(intersection, 'length'):
                            green_length += intersection.length
                        elif hasattr(intersection, 'geoms'):
                            # Handle MultiLineString or GeometryCollection
                            for geom in intersection.geoms:
                                if hasattr(geom, 'length'):
                                    green_length += geom.length
                
                except Exception as e:
                    # Skip problematic geometries
                    continue
            
            # Calculate percentage
            if total_path_length > 0:
                # Convert from degrees to approximate meters for coordinate systems
                # This is a rough approximation - for precise calculations, you'd need proper projection
                green_length_meters = green_length * 111000  # Rough conversion from degrees to meters
                percentage = min(100.0, (green_length_meters / total_path_length) * 100)
                return percentage
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error calculating green area coverage: {e}")
            return 0.0

    def _select_path_with_llm(self, path_options, from_node, to_node):
        """
        Use LLM to score paths and select the best one.
        Delegates to LLM interaction layer to avoid code duplication.
        
        Args:
            path_options: List of path dictionaries with metadata
            from_node: Starting node ID
            to_node: Destination node ID
            
        Returns:
            Selected path nodes or None if LLM fails
        """
        if not self.model.llm_enabled or not path_options:
            # Fallback to shortest path when LLM disabled
            return self._fallback_path_selection(path_options)
        
        try:
            # Prepare context for LLM
            context = {
                'time_of_day': self.model.hour_of_day,
                'from_node': from_node,
                'to_node': to_node
            }
            
            # Create agent state for LLM
            agent_state = {
                'age': getattr(self, 'age', 30),
                'current_needs': getattr(self, 'needs', {}),
                'agent_id': str(self.unique_id)
            }
            
            # Delegate to LLM interaction layer
            response = self.model.llm_interaction_layer.score_path_options(
                agent_state, path_options, context
            )
            
            # Extract selected path
            if 0 <= response.selected_path_id < len(path_options):
                selected_path = path_options[response.selected_path_id]
                self.logger.info(f"LLM selected path {response.selected_path_id + 1}: {response.reasoning}")
                return selected_path['path_nodes']
            else:
                return self._fallback_path_selection(path_options)
                
        except Exception as e:
            self.logger.error(f"Error in LLM path selection: {e}")
            return self._fallback_path_selection(path_options)

    def _fallback_path_selection(self, path_options):
        """
        Fallback path selection - choose shortest time.
        
        Args:
            path_options: Available path options
            
        Returns:
            Path nodes of shortest time path or None
        """
        if not path_options:
            return None
        
        # Select path with shortest travel time
        shortest_path = min(path_options, key=lambda p: p['travel_time_minutes'])
        return shortest_path['path_nodes']

    def _paths_are_same(self, path1, path2):
        """
        Check if two paths are essentially the same.
        
        Args:
            path1: First path dictionary
            path2: Second path dictionary
            
        Returns:
            Boolean indicating if paths are the same
        """
        if not path1 or not path2:
            return False
            
        return path1['path_nodes'] == path2['path_nodes']

    def _calculate_time_for_path(self, path_data):
        """
        Calculate travel time for a specific path.
        
        Args:
            path_data: Path dictionary with characteristics
            
        Returns:
            Travel time in minutes
        """
        return self._calculate_time_for_path_nodes(path_data['path_nodes'])

    def _calculate_time_for_path_nodes(self, path_nodes):
        """
        Calculate travel time for a list of path nodes.
        
        Args:
            path_nodes: List of node IDs in the path
            
        Returns:
            Travel time in minutes
        """
        total_distance = 0
        
        for i in range(len(path_nodes) - 1):
            edge_data = self.model.graph.get_edge_data(path_nodes[i], path_nodes[i + 1], {})
            edge_length = edge_data.get('length', 100)  # Default 100m if no data
            total_distance += edge_length
        
        # Use agent's step size to calculate time
        is_elderly = self.age >= 65
        step_size = 60.0 if is_elderly else 80.0
        
        return max(1, math.ceil(total_distance / step_size))
