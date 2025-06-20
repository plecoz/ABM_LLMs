from mesa.agent import Agent
from mesa_geo.geoagent import GeoAgent
from shapely.geometry import Point
import logging

class BaseAgent(GeoAgent):
    def __init__(self, model, unique_id, geometry, crs=None, **kwargs):
        """
        Initialize a base agent with modular properties.
    
        Args:
            model: Model instance the agent belongs to
            unique_id: Unique identifier for the agent
            geometry: Shapely geometry object
            crs: Coordinate reference system (included in kwargs if needed)
            **kwargs: Additional agent properties that can be customized
        """
        # Initialize GeoAgent (parent class) with correct order for Mesa-geo
        # Mesa-geo expects (model, geometry, crs) in that order
        super().__init__(model, geometry, crs)
        self.unique_id = unique_id
        
        # Basic agent properties with defaults that can be overridden
        self.age = kwargs.get('age', 30)
        self.gender = kwargs.get('gender', 'unspecified')
        self.income = kwargs.get('income', 50000)
        self.education = kwargs.get('education', 'high_school')
        self.occupation = kwargs.get('occupation', 'unspecified')
        self.home_location = kwargs.get('home_location', geometry)
        #self.current_node = home_node
        self.work_location = kwargs.get('work_location', None)
    
        # Movement and activity related attributes
        self.current_activity = None
        self.planned_activities = []
        self.movement_path = []
        self.speed = kwargs.get('speed', 1.0)  # movement speed in arbitrary units
        self.social_propensity = kwargs.get('social_propensity', 0.5)  # Likelihood to initiate social interaction
    
        # Communication and social network related attributes
        self.contacts = set()
        self.message_history = []
        self.online_contacts = set()
    
        # Decision module - can be replaced with custom logic
        self.decision_module = kwargs.get('decision_module', None)
    
        # Reference to the features module for scenario-specific properties
        self.features = kwargs.get('features', {})
    
        # History tracking
        self.location_history = []
        self.interaction_history = []
        # Initialize logger
        self.logger = logging.getLogger(f"Agent-{unique_id}")

    def step(self):
        """
        Advance the agent one step in the simulation.
        """
        # Update current position in location history
        # Use step_count from model
        step_count = getattr(self.model, 'step_count', 0)
        
        self.location_history.append((step_count, self.geometry))
        
        # Make decisions about next actions
        self._decide_next_action() 
        
        # Interact with other agents in the current location
        self._check_for_social_interaction()
        

    def _decide_next_action(self):
        """
        Determine the agent's next action.
        This uses the decision module if available, otherwise uses default behavior.
        """
        if self.decision_module:
            action = self.decision_module.decide_next_action(self, self.model)
            if action:
                self.planned_activities.append(action)
    
    def _check_for_social_interaction(self):
        """
        Check for and execute social interactions with other agents at the same location.
        This method is a placeholder to be implemented by subclasses, as the logic
        depends on how the model tracks co-located agents (e.g., at a POI).
        """
        pass
    
    def _interact(self):
        """
        DEPRECATED: This method is too generic.
        Interaction logic is now handled by `_check_for_social_interaction`
        which is triggered by specific contexts (e.g., waiting at a POI).
        """
        pass
            
    def _communicate_with(self, other_agent, online=False):
        """
        Communicate with another agent.
        
        Args:
            other_agent: The agent to communicate with
            online: Whether this is an online or offline interaction
        """
        # Get step count from model.step_count
        step_count = getattr(self.model, 'step_count', 0)
        
        # Generate a simple random message (can be replaced with LLM-generated content)
        message_content = f"Message from {self.unique_id} to {other_agent.unique_id} at step {step_count}"
        
        # Create message object
        message = {
            'sender': self.unique_id,
            'receiver': other_agent.unique_id,
            'content': message_content,
            'timestamp': step_count,
            'online': online
        }
        
        # Record in both agents' history
        self.message_history.append(message)
        other_agent.message_history.append(message)
        
        # Record in model's global communication history
        if hasattr(self.model, 'record_communication'):
            self.model.record_communication(message)
        
        # Add to interaction history
        interaction = {
            'type': 'communication',
            'agent_ids': [self.unique_id, other_agent.unique_id],
            'online': online,
            'timestamp': step_count,
            'content': message_content
        }
        
        self.interaction_history.append(interaction)
        other_agent.interaction_history.append(interaction)
    

    
    def get_history(self):
        """
        Return the agent's history data.
        """
        return {
            'location_history': self.location_history,
            'interaction_history': self.interaction_history,
            'message_history': self.message_history
        }
    
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