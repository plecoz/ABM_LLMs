from mesa.agent import Agent
from agents.base_agent import BaseAgent
import random
import networkx as nx

class Resident(BaseAgent):
    def __init__(self, model, unique_id, geometry, home_node, accessible_nodes, **kwargs):
        """
        Proper initialization for Mesa 3.x:
        - model MUST be the first argument to parent class
        - unique_id is set separately
        """
        # Parent class gets model only
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
        
        # Health attributes
        self.health_status = kwargs.get('health_status', 'healthy')
        
        # Register with model (automatic in Mesa, but explicit here)
        #model.register_agent(self)

    def move_to_poi(self, poi_type):
        """Improved movement with distance check"""
        if not self.model.pois.get(poi_type):
            return False
        
        valid_pois = [n for n in self.model.pois[poi_type] 
                     if n in self.accessible_nodes]
        
        if not valid_pois:
            return False
            
        try:
           #target = random.choice(self.model.pois[poi_type])
            target = random.choice(valid_pois)
            """
            distance = nx.shortest_path_length(
                self.model.graph,
                self.current_node,
                target,
                weight="length"
            )
            if distance <= 1000:  # 15-min walk threshold (~1km)
                self.current_node = target
                self.visited_pois.append(target)
                return True
            """
            self.current_node = target
            self.visited_pois.append(target)
        except (nx.NetworkXNoPath, KeyError):
            pass
        return False

    def step(self):
        super().step()
        if self.model.pois:
            self.move_to_poi(random.choice(list(self.model.pois.keys())))
        self._update_health_status()
        self._maintain_social_network()


    def _update_health_status(self):
        """
        Update the agent's health status based on interactions and environment.
        This is a placeholder for more complex health models.
        """
        # Check if we have scenario-specific health features
        if 'health' in self.features:
            health_features = self.features['health']
            
            # Example: Simple infectious disease logic
            if health_features.get('infectious_disease_enabled', False):
                if self.health_status == 'infected':
                    # Recovery chance
                    recovery_probability = health_features.get('recovery_probability', 0.1)
                    if self.model.random.random() < recovery_probability:
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
            
            # Check if model has get_agent_by_id method
            if hasattr(self.model, 'get_agent_by_id'):
                contact_agent = self.model.get_agent_by_id(contact_id)
                
                if contact_agent:
                    # Random choice between online and offline communication
                    online = self.model.random.random() < 0.7  # 70% chance for online
                    self._communicate_with(contact_agent, online=online)
