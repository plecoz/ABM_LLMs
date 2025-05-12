
from agents.base_agent import BaseAgent
from agents.resident import Resident

class OrganizationAgent(BaseAgent):
    """
    Organization agent representing entities like government, schools, or businesses.
    This class extends BaseAgent with organization-specific attributes and behaviors.
    """
    
    def __init__(self, unique_id, model, geometry, **kwargs):
        """
        Initialize an organization agent.
        
        Args:
            unique_id: Unique identifier for the agent
            model: Model instance the agent belongs to
            geometry: Shapely geometry representing the agent's location
            **kwargs: Additional agent properties that can be customized
        """
        super().__init__(unique_id, model, geometry, **kwargs)
        
        # Organization-specific attributes
        self.org_type = kwargs.get('org_type', 'business')  # business, government, school, etc.
        self.capacity = kwargs.get('capacity', 100)  # how many people can be here
        self.employees = kwargs.get('employees', [])
        self.visitors = set()  # current visitors
        self.open_hours = kwargs.get('open_hours', {'start': 9, 'end': 17})
        self.services = kwargs.get('services', [])
        
        # Policy and influence attributes
        self.influence_radius = kwargs.get('influence_radius', 1.0)  # geographical radius of influence
        self.policies = kwargs.get('policies', {})
        
        # For government organizations
        self.jurisdiction = kwargs.get('jurisdiction', None)
        
        # For businesses
        self.products = kwargs.get('products', [])
        self.prices = kwargs.get('prices', {})
        
        # For schools
        self.education_level = kwargs.get('education_level', None)
        self.students = kwargs.get('students', [])
    
    def step(self):
        """
        Advance the organization agent one step in the simulation.
        """
        # Call the parent class step method for basic behavior
        super().step()
        
        # Organization-specific behavior
        self._update_visitors()
        self._implement_policies()
    
    def _update_visitors(self):
        """
        Update the set of visitors at this organization.
        """
        # Clear previous visitors
        self.visitors = set()
        """
        # Get agents at this location
        nearby_agents = self.model.get_nearby_agents(self, distance=0.01)
        
        for agent in nearby_agents:
            if isinstance(agent, Resident):
                self.visitors.add(agent.unique_id)
                
        """
    
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
"""    
    def add_employee(self, agent_id):
        
        Add an employee to this organization.
        
        Args:
            agent_id: ID of the agent to add as employee
        
        if agent_id not in self.employees:
            self.employees.append(agent_id)
            
            # Also update the agent if it exists
            agent = self.model.get_agent_by_id(agent_id)
            if agent and isinstance(agent, Resident):
                agent.occupation = f"Employee at {self.unique_id} ({self.org_type})"
                agent.work_location = self.geometry
    
    def remove_employee(self, agent_id):
        
        Remove an employee from this organization.
        
        Args:
            agent_id: ID of the agent to remove
        
        if agent_id in self.employees:
            self.employees.remove(agent_id)
            
            # Update the agent if it exists
            agent = self.model.get_agent_by_id(agent_id)
            if agent and isinstance(agent, Resident) and agent.work_location == self.geometry:
                agent.occupation = "unemployed"
                agent.work_location = None
    
    def set_policy(self, policy_name, config):
        
        Set or update a policy for this organization.
        
        Args:
            policy_name: Name of the policy
            config: Policy configuration dictionary
        
        self.policies[policy_name] = config
"""