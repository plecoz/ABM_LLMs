from mesa.agent import Agent
from agents.base_agent import BaseAgent
import random
import networkx as nx
import logging

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
        
        # Health attributes
        self.health_status = kwargs.get('health_status', 'healthy')
        
        # Initialize logger if not provided
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"Resident-{unique_id}")

    def move_to_poi(self, poi_type):
        """Improved movement with distance check"""
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
            self.current_node = target
            self.visited_pois.append(target)
            return True
        except (nx.NetworkXNoPath, KeyError, IndexError) as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error moving to POI: {e}")
        return False

    def step(self):
        """Advance the agent one step"""
        try:
            super().step()
            if hasattr(self.model, 'pois') and self.model.pois:
                self.move_to_poi(random.choice(list(self.model.pois.keys())))
            self._update_health_status()
            self._maintain_social_network()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in resident step: {e}")

    def _update_health_status(self):
        """
        Update the agent's health status based on interactions and environment.
        This is a placeholder for more complex health models.
        """
        # Check if we have scenario-specific health features
        if hasattr(self, 'features') and 'health' in self.features:
            health_features = self.features['health']
            
            # Example: Simple infectious disease logic
            if health_features.get('infectious_disease_enabled', False):
                if self.health_status == 'infected':
                    # Recovery chance
                    recovery_probability = health_features.get('recovery_probability', 0.1)
                    if hasattr(self.model, 'random') and self.model.random.random() < recovery_probability:
                        self.health_status = 'recovered'
                        # Get step count safely
                        step_count = getattr(self.model, 'schedule', None)
                        step_count = step_count.steps if step_count is not None else 0
                        self.logger.info(f"Agent {self.unique_id} recovered at step {step_count}")
    
    def _maintain_social_network(self):
        """
        Maintain the agent's social network through regular contact.
        """
        # Random chance to communicate with someone in social network
        if self.social_network and hasattr(self.model, 'random') and self.model.random.random() < 0.1:  # 10% chance
            contact_id = self.model.random.choice(self.social_network)
            
            # Get the contact agent
            contact_agent = self.model.get_agent_by_id(contact_id)
            
            if contact_agent:
                # Random choice between online and offline communication
                online = self.model.random.random() < 0.7  # 70% chance for online
                # Use the _communicate_with method from BaseAgent
                if hasattr(self, '_communicate_with'):
                    self._communicate_with(contact_agent, online=online)
    
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
