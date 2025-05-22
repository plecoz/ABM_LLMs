from agents.base_agent import BaseAgent
from shapely.geometry import Point
import logging

class POI(BaseAgent):
    """
    Point of Interest (POI) agent representing locations like shops, hospitals, schools, etc.
    
    A POI is a static agent (doesn't move) but can interact with residents and organizations.
    """
    
    def __init__(self, model, unique_id, geometry, node_id, poi_type, **kwargs):
        """
        Initialize a POI agent.
        
        Args:
            model: Model instance the agent belongs to
            unique_id: Unique identifier for the agent
            geometry: Shapely geometry object representing the agent's location
            node_id: The node ID in the street network where this POI is located
            poi_type: Type of POI (e.g., 'hospital', 'school', 'supermarket')
            **kwargs: Additional agent properties that can be customized
        """
        super().__init__(model, unique_id, geometry, **kwargs)
        
        self.node_id = node_id
        self.poi_type = poi_type
        self.category = kwargs.get('category', self._determine_category(poi_type))
        self.visitors = set()  # Set of agent IDs currently visiting
        self.capacity = kwargs.get('capacity', 50)
        self.open_hours = kwargs.get('open_hours', {'start': 8, 'end': 20})  # Default 8am-8pm
        
        # POI-specific attributes
        self.service_quality = kwargs.get('service_quality', 3)  # 1-5 scale
        self.popularity = kwargs.get('popularity', 0)  # Increases as more agents visit
        
        # Initialize logger if not provided
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"POI-{unique_id}")
    
    def _determine_category(self, poi_type):
        """
        Determine the category of a POI based on its type.
        
        Args:
            poi_type: Type of POI
            
        Returns:
            Category string
        """
        healthcare_types = ['hospital', 'clinic', 'pharmacy']
        education_types = ['school', 'university', 'library', 'kindergarten']
        shopping_types = ['supermarket', 'convenience', 'mall', 'department_store', 'grocery']
        recreation_types = ['park', 'sports_centre', 'garden', 'playground', 'fitness_centre']
        services_types = ['bank', 'police', 'fire_station', 'post_office', 'government']
        food_types = ['restaurant', 'cafe']
        
        if poi_type in healthcare_types:
            return 'healthcare'
        elif poi_type in education_types:
            return 'education'
        elif poi_type in shopping_types:
            return 'shopping'
        elif poi_type in recreation_types:
            return 'recreation'
        elif poi_type in services_types:
            return 'services'
        elif poi_type in food_types:
            return 'food'
        else:
            return 'other'
    
    def step(self):
        """
        POIs are mostly static but can update visitor information and popularity.
        """
        # Update visitors
        self._update_visitors()
        
        # Update popularity based on visitor count
        if len(self.visitors) > 0:
            # Increment popularity (with a cap)
            self.popularity = min(5, self.popularity + 0.01 * len(self.visitors))
    
    def _update_visitors(self):
        """
        Update the set of visitors at this POI.
        """
        # Clear previous visitors
        self.visitors = set()
        
        # Get agents at this location's node
        if hasattr(self.model, 'residents'):
            for resident in self.model.residents:
                if hasattr(resident, 'current_node') and resident.current_node == self.node_id:
                    self.visitors.add(resident.unique_id)
                # Also check for residents who have moved directly to this POI's location
                elif hasattr(resident, 'geometry') and self.geometry.distance(resident.geometry) < 0.001:
                    self.visitors.add(resident.unique_id)
    
    def is_open(self, hour):
        """
        Check if the POI is open at the given hour.
        
        Args:
            hour: Current hour (0-23)
            
        Returns:
            Boolean indicating if the POI is open
        """
        return self.open_hours['start'] <= hour < self.open_hours['end']
    
    def get_info(self):
        """
        Get information about this POI.
        
        Returns:
            Dictionary with POI details
        """
        return {
            "id": self.unique_id,
            "type": self.poi_type,
            "category": self.category,
            "node_id": self.node_id,
            "location": (self.geometry.x, self.geometry.y),
            "open_hours": self.open_hours,
            "capacity": self.capacity,
            "current_visitors": len(self.visitors),
            "popularity": self.popularity,
            "service_quality": self.service_quality
        } 