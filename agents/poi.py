from agents.base_agent import BaseAgent
from shapely.geometry import Point
import logging
import random

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
        
        # If category is explicitly provided, use it, otherwise determine it from poi_type
        if 'category' in kwargs:
            self.category = kwargs['category']
            print(f"Using provided category for POI {unique_id}: {self.category}")
        else:
            self.category = self._determine_category(poi_type)
        
        self.visitors = set()  # Set of agent IDs currently visiting
        self.capacity = kwargs.get('capacity', 50)
        self.open_hours = kwargs.get('open_hours', {'start': 8, 'end': 20})  # Default 8am-8pm
        
        # POI-specific attributes
        #self.service_quality = kwargs.get('service_quality', 3)  # 1-5 scale
        #self.popularity = kwargs.get('popularity', 0)  # Increases as more agents visit
        
        # Waiting time attributes
        self.has_waiting_time = self._has_waiting_time()
        self.base_service_time = self._get_base_service_time()  # Minutes per customer
        self.current_waiting_time = 0  # Current waiting time in minutes
        
        # Peak hour definitions (24-hour format)
        self.peak_hours = {
            'morning': (7, 9),    # 7-9 AM
            'lunch': (12, 14),    # 12-2 PM
            'evening': (17, 19)   # 5-7 PM
        }
        
        # Initialize logger if not provided
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"POI-{unique_id}")
    
    def _has_waiting_time(self):
        """
        Determine if this POI type should have waiting times.
        
        Returns:
            Boolean indicating if this POI should have waiting times
        """
        # POI types that have waiting times: grocery stores, banks, restaurants, barber shops, pharmacies
        waiting_time_types = [
            'supermarket', 'grocery', 'bank', 'restaurant', 'cafe', 
            'barber', 'pharmacy', 'convenience', 'marketplace'
        ]
        
        return any(wt_type in self.poi_type.lower() for wt_type in waiting_time_types)
    
    def _get_base_service_time(self):
        """
        Get the base service time in minutes for this POI type.
        
        Returns:
            Base service time in minutes
        """
        # Set all POI types to have 5 minutes service time
        return 30
    
    def _determine_category(self, poi_type):
        """
        Determine the category of a POI based on its type.
        
        Args:
            poi_type: Type of POI
            
        Returns:
            Category string
        """
        # Daily Living: Grocery stores, banks, restaurants, barber shops, post offices
        daily_living_types = [
            'supermarket', 'convenience', 'grocery', 'bank', 'restaurant', 
            'cafe', 'barber', 'post_office', 'atm', 'marketplace', 'bakery', 
            'butcher', 'laundry', 'convenience_store'
        ]
        
        # Healthcare: Hospitals, clinics, pharmacies
        healthcare_types = [
            'hospital', 'clinic', 'pharmacy', 'doctor', 'dentist', 
            'healthcare', 'medical_center', 'emergency'
        ]
        
        # Education: Kindergartens, primary schools, and secondary schools
        education_types = [
            'school', 'kindergarten', 'primary_school', 'secondary_school',
            'high_school', 'university', 'college', 'educational_institution'
        ]
        
        # Entertainment: Parks, public squares, libraries, museums, art galleries, 
        # cultural centers, theaters, gyms, and stadiums
        entertainment_types = [
            'park', 'square', 'public_square', 'library', 'museum', 'gallery', 'art_gallery',
            'cultural_center', 'theater', 'cinema', 'gym', 'fitness_centre', 
            'sports_centre', 'stadium', 'recreation', 'playground', 'garden', 
            'entertainment', 'community_center'
        ]
        
        # Public Transportation: Bus stops
        transportation_types = [
            'bus_stop', 'bus_station', 'transit_stop', 'public_transport',
            'transport', 'station', 'transit', 'platform', 'stop_position'
        ]
        
        # Print debugging info
        print(f"Determining category for POI type: {poi_type}")
        
        # Exact match check first
        if poi_type.lower() in daily_living_types:
            category = 'daily_living'
        elif poi_type.lower() in healthcare_types:
            category = 'healthcare'
        elif poi_type.lower() in education_types:
            category = 'education'
        elif poi_type.lower() in entertainment_types:
            category = 'entertainment'
        elif poi_type.lower() in transportation_types:
            category = 'transportation'
        # If no exact match, check for partial matches
        elif any(t in poi_type.lower() for t in ['bus', 'stop', 'station', 'transit']):
            category = 'transportation'
        elif any(t in poi_type.lower() for t in ['hospital', 'clinic', 'doctor', 'pharmacy']):
            category = 'healthcare'
        elif any(t in poi_type.lower() for t in ['school', 'education', 'university', 'college']):
            category = 'education'
        elif any(t in poi_type.lower() for t in ['park', 'museum', 'theater', 'library', 'recreation']):
            category = 'entertainment'
        elif any(t in poi_type.lower() for t in ['market', 'shop', 'store', 'bank', 'restaurant']):
            category = 'daily_living'
        else:
            category = 'other'
            
        print(f"  Result: {poi_type} → {category}")
        return category
    
    def step(self):
        """
        POIs are mostly static but can update visitor information and popularity.
        """
        # Update visitors
        self._update_visitors()
        
        # Update waiting time
        self._update_waiting_time()
        
        # Update popularity based on visitor count
        #POIs are mostly static but can update visitor information and popularity.
        """
        if len(self.visitors) > 0:
            # Increment popularity (with a cap)
            self.popularity = min(5, self.popularity + 0.01 * len(self.visitors))
        POIs are mostly static but can update visitor information and popularity.
        """
    
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
    
    def _update_waiting_time(self):
        """
        Update the waiting time based on time of day and number of visitors.
        """
        if not self.has_waiting_time:
            self.current_waiting_time = 0
            return
            
        # Get current hour from model
        current_hour = self.model.hour_of_day if hasattr(self.model, 'hour_of_day') else 12
        
        # Check if current hour is a peak hour
        is_peak_hour = False
        for peak_period, (start, end) in self.peak_hours.items():
            if start <= current_hour < end:
                is_peak_hour = True
                break
        
        # Calculate base waiting time based on visitors
        visitor_count = len(self.visitors)
        
        # Calculate waiting time (minutes)
        if visitor_count == 0:
            self.current_waiting_time = 0
        else:
            # Base calculation: service time * number of visitors / capacity
            base_wait = self.base_service_time * visitor_count / max(1, self.capacity)
            
            # Apply peak hour multiplier if needed
            peak_multiplier = 1.5 if is_peak_hour else 1.0
            
            # Add some randomness (±20%)
            randomness = random.uniform(0.8, 1.2)
            
            self.current_waiting_time = base_wait * peak_multiplier * randomness
    
    def is_open(self, hour):
        """
        Check if the POI is open at the given hour.
        
        Args:
            hour: Current hour (0-23)
            
        Returns:
            Boolean indicating if the POI is open
        """
        return self.open_hours['start'] <= hour < self.open_hours['end']
    
    def get_waiting_time(self):
        """
        Get the current waiting time in minutes.
        
        Returns:
            Current waiting time in minutes, or 0 if no waiting time applies
        """
        if not self.has_waiting_time:
            return 0
        
        #return max(0, round(self.current_waiting_time))
        return self.base_service_time
    
    def get_info(self):
        """
        Get information about this POI.
        
        Returns:
            Dictionary with POI details
        """
        info = {
            "id": self.unique_id,
            "type": self.poi_type,
            "category": self.category,
            "node_id": self.node_id,
            "location": (self.geometry.x, self.geometry.y),
            "open_hours": self.open_hours,
            "capacity": self.capacity,
            "current_visitors": len(self.visitors),
            #"popularity": self.popularity,
            #"service_quality": self.service_quality
        }
        
        # Add waiting time information if applicable
        if self.has_waiting_time:
            info["waiting_time"] = self.get_waiting_time()
            info["base_service_time"] = self.base_service_time
        
        return info 