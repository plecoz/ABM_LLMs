"""
Configuration file for Points of Interest (POIs) in the simulation.
Edit this file to control which POIs are included in the simulation.
"""

# List of POI types to include in the simulation
# Comment out or remove POI types you don't want to include
SELECTED_POIS = [
    # Essential services
    'bank',
    'police',
    'fire_station',
    
    # Education
    'school',
    'university',
    'library',
    
    # Healthcare
    'hospital',
    'clinic',
    'pharmacy',
    
    # Other amenities - comment these out if you only want the essential services
    # 'restaurant',
    # 'cafe',
    # 'supermarket',
    # 'convenience',
    # 'post_office',
    # 'park',
    # 'sports_centre',
    # 'garden',
    # 'playground',
]

# Use only specific POIs (banks, police, schools, hospitals, fire stations)
ESSENTIAL_SERVICES_ONLY = [
    'bank',
    'police',
    'school',
    'hospital',
    'fire_station',
]

# Function to get the active POI configuration
def get_active_poi_config():
    """
    Returns the currently active POI configuration.
    Edit this function to switch between different configurations.
    """
    # Return SELECTED_POIS for all configured POIs
    # Return ESSENTIAL_SERVICES_ONLY for only essential services
    # Return None to include all available POIs
    
    # CHANGE THIS LINE to control which POIs are included:
    return ESSENTIAL_SERVICES_ONLY 