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
        
        # Parish information
        self.parish = kwargs.get('parish', None)
        
        # Demographic attributes from model (may vary by parish)
        self.age = kwargs.get('age', 30)  # Default age
        self.gender = kwargs.get('gender', 'male')  # Default gender
        self.income = kwargs.get('income', 50000)  # Default income
        self.education = kwargs.get('education', 'high_school')  # Default education level
        
        # Initialize logger if not provided
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"Resident-{unique_id}")

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
        # First try to find POI agents of the specified type
        poi_agents = [agent for agent in self.model.poi_agents if agent.poi_type == poi_type]
        
        if poi_agents:
            # Filter POIs to only those at accessible nodes
            accessible_pois = [poi for poi in poi_agents if poi.node_id in self.accessible_nodes]
            
            if accessible_pois:
                # Choose a random accessible POI
                target_poi = random.choice(accessible_pois)
                self.current_node = target_poi.node_id
                self.geometry = target_poi.geometry  # Update geometry to POI location
                self.visited_pois.append(target_poi.node_id)
                
                # Add resident to POI's visitors
                if hasattr(target_poi, 'visitors'):
                    target_poi.visitors.add(self.unique_id)
                    
                return True
        
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
            self.current_node = target
            self.visited_pois.append(target)
            
            # Update geometry based on the node coordinates
            node_coords = self.model.graph.nodes[target]
            if 'x' in node_coords and 'y' in node_coords:
                from shapely.geometry import Point
                self.geometry = Point(node_coords['x'], node_coords['y'])
                
            return True
        except (nx.NetworkXNoPath, KeyError, IndexError) as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error moving to POI: {e}")
        return False

    def step(self):
        """Advance the agent one step"""
        try:
            super().step()
            
            # Try to move to a POI agent
            if hasattr(self.model, 'poi_agents') and self.model.poi_agents:
                # Get all unique POI types
                poi_types = list(set(poi.poi_type for poi in self.model.poi_agents))
                if poi_types:
                    self.move_to_poi(random.choice(poi_types))
            # Fall back to old method if POI agents aren't available
            elif hasattr(self.model, 'pois') and self.model.pois:
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
                "education": self.education
            }
        }
