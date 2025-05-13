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
        
        # Move along planned path if available
        self._move()
        
        # Interact with other agents
        self._interact()
        
        # Execute current activity
        self._perform_activity()
        

    def _decide_next_action(self):
        """
        Determine the agent's next action.
        This uses the decision module if available, otherwise uses default behavior.
        """
        if self.decision_module:
            action = self.decision_module.decide_next_action(self, self.model)
            if action:
                self.planned_activities.append(action)
        elif hasattr(self.model, 'random'):
            # Default simple random behavior
            if not self.planned_activities and self.model.random.random() < 0.2:  # 20% chance
                # 20% chance to add a new random activity
                poi = self._select_random_poi()
                if poi:
                    self.planned_activities.append({
                        'type': 'visit',
                        'location': poi,
                        'duration': self.model.random.randint(1, 5)
                    })
    
    def _move(self):
        """
        Move the agent along its planned path.
        """
        # If we have a movement path, follow it
        if self.movement_path:
            next_point = self.movement_path.pop(0)
            self.geometry = Point(next_point)
        # If we need to go somewhere but don't have a path yet
        elif self.planned_activities and 'location' in self.planned_activities[0]:
            destination = self.planned_activities[0]['location']
            self._plan_route(destination)
    
    def _plan_route(self, destination):
        """
        Plan a route to the destination using Google Maps API or simple path.
        In this initial implementation, we'll use a simplified straight path.
        """
        if hasattr(self.model, 'route_planner') and self.model.route_planner:
            # Use the model's route planner (could be Google Maps API wrapper)
            self.movement_path = self.model.route_planner.get_route(
                origin=self.geometry, 
                destination=destination
            )
        else:
            # Simple linear path as fallback
            start_x, start_y = self.geometry.x, self.geometry.y
            end_x, end_y = destination.x, destination.y
            
            # Create a simple path with 5 points
            steps = 5
            self.movement_path = [
                Point(
                    start_x + (end_x - start_x) * i / steps,
                    start_y + (end_y - start_y) * i / steps
                )
                for i in range(1, steps + 1)
            ]
    
    def _interact(self):
        """
        Interact with nearby agents.
        """
        # Check if the model has the get_nearby_agents method
        if not hasattr(self.model, 'get_nearby_agents'):
            return
            
        # Find nearby agents
        nearby_agents = self.model.get_nearby_agents(self)
        
        for agent in nearby_agents:
            # Add to contacts
            self.contacts.add(agent.unique_id)
            agent.contacts.add(self.unique_id)
            
            # Random chance to communicate (can be replaced with more complex logic)
            if hasattr(self.model, 'random') and self.model.random.random() < 0.3:  # 30% chance
                self._communicate_with(agent, online=False)
    
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
    
    def _perform_activity(self):
        """
        Perform the current activity if one exists.
        """
        if not self.planned_activities:
            return
            
        current = self.planned_activities[0]
        
        # If we've reached the activity location, perform it
        if 'location' in current:
            target_location = current['location']
            
            # Check if we've reached the location (within a small distance)
            if self.geometry.distance(target_location) < 0.001:
                # We're at the location, reduce the duration
                if 'duration' in current:
                    current['duration'] -= 1
                    
                    # If the activity is complete, remove it
                    if current['duration'] <= 0:
                        self.planned_activities.pop(0)
                        self.current_activity = None
                    else:
                        self.current_activity = current
                else:
                    # No duration specified, complete immediately
                    self.planned_activities.pop(0)
                    self.current_activity = None
    
    def _select_random_poi(self):
        """
        Select a random point of interest to visit.
        """
        if not hasattr(self.model, 'random'):
            return None
            
        if hasattr(self.model, 'poi_selector') and self.model.poi_selector:
            # Use the model's POI selector if available
            return self.model.poi_selector.select_poi(self)
        elif hasattr(self.model, 'points_of_interest') and self.model.points_of_interest:
            # Fallback to simple random selection from model's POIs
            return self.model.random.choice(self.model.points_of_interest)
        
        return None
    
    def get_history(self):
        """
        Return the agent's history data.
        """
        return {
            'location_history': self.location_history,
            'interaction_history': self.interaction_history,
            'message_history': self.message_history
        }