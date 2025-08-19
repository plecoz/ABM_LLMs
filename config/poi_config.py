"""
Configuration file for Points of Interest (POIs) in the simulation.
Edit this file to control which POIs are included in the simulation.
"""

# POI Configuration File

# All POI types by category
POI_TYPES = {
    "daily_living": [
        'supermarket', 'convenience', 'grocery', 'bank', 'restaurant', 
        'cafe', 'barber', 'post_office', 'atm', 'marketplace', 'bakery', 
        'butcher', 'laundry', 'convenience_store'
    ],
    "healthcare": [
        'hospital', 'clinic', 'pharmacy', 'doctor', 'dentist', 
        'healthcare', 'medical_center', 'emergency'
    ],
    "education": [
        'school', 'kindergarten', 'primary_school', 'secondary_school',
        'high_school', 'university', 'college', 'educational_institution'
    ],
    "entertainment": [
        'park', 'square', 'public_square', 'library', 'museum', 'gallery', 'art_gallery',
        'cultural_center', 'theater', 'cinema', 'gym', 'fitness_centre', 
        'sports_centre', 'stadium', 'recreation', 'playground', 'garden', 
        'entertainment', 'community_center'
    ],
    "transportation": [
        'bus_stop', 'bus_station', 'transit_stop', 'public_transport',
        'transport', 'station', 'transit', 'platform', 'stop_position'
    ],
    "casino": [
        'casino', 'gambling', 'gaming'
    ]
}

# Essential services only (subset of all POIs)
ESSENTIAL_SERVICES_ONLY = [
    'supermarket', 'grocery', 'bank', 'hospital', 'clinic', 'pharmacy',
    'school', 'bus_stop', 'bus_station'
]

# Function to get the active POI configuration
def get_active_poi_config():
    """
    Returns the list of POI types to use in the simulation.
    By default, returns all POI types.
    
    Returns:
        List of POI types or None to use all available POIs
    """
    # Return None to use all POIs by default
    return None 