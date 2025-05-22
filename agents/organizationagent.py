from agents.base_agent import BaseAgent
from agents.resident import Resident

class OrganizationAgent(BaseAgent):
    """
    Organization agent representing businesses, government agencies, schools, etc.
    """
    def __init__(self, model, unique_id, geometry, current_node, org_type='business', **kwargs):
        """
        Initialize an organization agent.
        
        Args:
            model: Model instance the agent belongs to
            unique_id: Unique identifier for the agent
            geometry: Shapely geometry object representing the agent's location
            current_node: The agent's current node in the network
            org_type: Type of organization (business, government, school, etc.)
            **kwargs: Additional agent properties that can be customized
        """
        super().__init__(model, unique_id, geometry, **kwargs)
        
        self.current_node = current_node
        self.org_type = org_type
        self.visitors = set()  # Set of agent IDs currently visiting
        self.policies = {}
        self.influence_radius = kwargs.get('influence_radius', 0.5)  # km
        
        # Organization-specific attributes
        self.capacity = kwargs.get('capacity', 100)
        self.open_hours = kwargs.get('open_hours', {'start': 8, 'end': 18})  # Default 8am-6pm
        
        # Parish information
        self.parish = kwargs.get('parish', None)
        
        # Cooling center attributes
        self.is_cooling_center = kwargs.get('is_cooling_center', False)
        self.has_ac = kwargs.get('has_ac', org_type in ['business', 'government', 'school'])
        self.cooling_capacity = kwargs.get('cooling_capacity', 50)  # How many people can cool off here
        
        # Initialize policies
        self._initialize_policies()
        
    def _initialize_policies(self):
        """
        Initialize organization policies.
        """
        # Default policies based on organization type
        if self.org_type == 'government':
            self.policies = {
                'information_broadcast': {
                    'active': False,
                    'message': "Public service announcement",
                    'online': True
                }
            }
        elif self.org_type == 'business':
            self.policies = {
                'information_broadcast': {
                    'active': False,
                    'message': "Business announcement",
                    'online': False
                }
            }
        elif self.org_type == 'school':
            self.policies = {
                'information_broadcast': {
                    'active': False,
                    'message': "School announcement",
                    'online': True
                }
            }
            
        # Add heatwave response policy for all organizations
        self.policies['heatwave_response'] = {
            'active': False,
            'cooling_center': self.is_cooling_center,
            'extended_hours': False
        }

    def step(self):
        """
        Advance the organization agent one step in the simulation.
        """
        # Call the parent class step method for basic behavior
        super().step()
        
        # Organization-specific behavior
        self._update_visitors()
        self._check_heatwave_conditions()
        self._implement_policies()
    
    def _update_visitors(self):
        """
        Update the set of visitors at this organization.
        """
        # Clear previous visitors
        self.visitors = set()
        
        # Get agents at this location
        nearby_agents = self.model.get_nearby_agents(self, distance=0.01)
        
        for agent in nearby_agents:
            if isinstance(agent, Resident):
                self.visitors.add(agent.unique_id)
                
    def _check_heatwave_conditions(self):
        """
        Check if there's a heatwave and activate cooling center if needed.
        """
        if not hasattr(self.model, 'environmental_conditions'):
            return
            
        env = self.model.environmental_conditions
        
        # Check if there's a heatwave
        if env.get('heatwave_active', False):
            # Activate cooling center policy if this organization has AC
            if self.has_ac:
                heatwave_policy = self.policies.get('heatwave_response', {})
                
                # Activate cooling center
                heatwave_policy['active'] = True
                
                # Government buildings and schools are more likely to become official cooling centers
                if self.org_type in ['government', 'school'] and not heatwave_policy.get('cooling_center', False):
                    if hasattr(self.model, 'random') and self.model.random.random() < 0.7:  # 70% chance
                        heatwave_policy['cooling_center'] = True
                        self.is_cooling_center = True
                        self.model.logger.info(f"Organization {self.unique_id} ({self.org_type}) activated as cooling center")
                
                # Extend hours during severe heatwave
                current_temp = self.model.get_current_temperature()
                if current_temp > 35 and not heatwave_policy.get('extended_hours', False):
                    heatwave_policy['extended_hours'] = True
                    self.open_hours['end'] = 22  # Extended to 10pm
                    self.model.logger.info(f"Organization {self.unique_id} extended hours until 10pm due to extreme heat")
        else:
            # Reset policies when heatwave ends
            heatwave_policy = self.policies.get('heatwave_response', {})
            if heatwave_policy.get('active', False):
                heatwave_policy['active'] = False
                heatwave_policy['extended_hours'] = False
                # Reset hours
                self.open_hours['end'] = 18  # Back to 6pm
    
    def _implement_policies(self):
        """
        Implement organization policies that affect agents.
        """
        if not self.policies:
            return
            
        # Get agents within influence radius
        influenced_agents = self.model.get_nearby_agents(self, distance=self.influence_radius)
        
        for policy, config in self.policies.items():
            if policy == 'information_broadcast' and config.get('active', False):
                # Broadcast information to nearby agents
                message = config.get('message', f"Announcement from {self.unique_id}")
                
                for agent in influenced_agents:
                    if isinstance(agent, Resident):
                        self._communicate_with(agent, online=config.get('online', False))
            
            elif policy == 'heatwave_response' and config.get('active', False):
                # Apply cooling effect to visitors
                self._provide_cooling_to_visitors()
    
    def _provide_cooling_to_visitors(self):
        """
        Provide cooling to visitors during a heatwave.
        """
        # Check if this is a cooling center or has AC
        if not self.has_ac:
            return
            
        cooling_strength = 10  # Base cooling effect
        
        # Official cooling centers provide better cooling
        if self.is_cooling_center:
            cooling_strength = 20
            
        # Apply cooling effect to visitors
        for visitor_id in self.visitors:
            visitor = self.model.get_agent_by_id(visitor_id)
            if visitor and hasattr(visitor, 'heat_stress_level'):
                # Reduce heat stress
                visitor.heat_stress_level = max(0, visitor.heat_stress_level - cooling_strength)
                
                # Log significant cooling
                if cooling_strength >= 15:
                    self.model.logger.info(f"Agent {visitor_id} cooled down at {self.org_type} cooling center {self.unique_id}")
    
    def activate_as_cooling_center(self):
        """
        Activate this organization as an official cooling center.
        """
        if self.has_ac:
            self.is_cooling_center = True
            
            # Update policy
            if 'heatwave_response' in self.policies:
                self.policies['heatwave_response']['cooling_center'] = True
                
            # Log activation
            self.model.logger.info(f"Organization {self.unique_id} ({self.org_type}) manually activated as cooling center")
            
            return True
        return False
        
    def get_parish_info(self):
        """
        Get information about the organization's parish.
        
        Returns:
            Dictionary with parish details or None if no parish assigned
        """
        if not self.parish:
            return None
            
        return {
            "parish_name": self.parish,
            "location": (self.geometry.x, self.geometry.y),
            "org_type": self.org_type,
            "capacity": self.capacity
        }
