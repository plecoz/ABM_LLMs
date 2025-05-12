from mesa.agent import Agent


class BaseAgent(Agent):
    def __init__(self, model, unique_id, **kwargs):
        """
        Initialize a base agent with modular properties.
    
        Args:
            model: Model instance the agent belongs to
            unique_id: Unique identifier for the agent
            **kwargs: Additional agent properties that can be customized
        """
        super().__init__(model)
        self.unique_id = unique_id
        
        # Basic agent properties with defaults that can be overridden
        self.age = kwargs.get('age', 30)
        self.gender = kwargs.get('gender', 'unspecified')
        self.income = kwargs.get('income', 50000)
        self.education = kwargs.get('education', 'high_school')
        self.occupation = kwargs.get('occupation', 'unspecified')
        #self.home_location = kwargs.get('home_location', home_node)
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

    def step(self):
        """
        Advance the agent one step in the simulation.
        """
        # Update current position in location history
        #self.location_history.append((self.model.schedule.steps, self.geometry))
        
        # Make decisions about next actions
        #self._decide_next_action()
        
        # Move along planned path if available
        #self._move()
        
        # Interact with other agents
        #self._interact()
        
        # Execute current activity
        #self._perform_activity()
        
        # Initialize logger
        #self.logger = logging.getLogger(f"Agent-{unique_id}")
        pass