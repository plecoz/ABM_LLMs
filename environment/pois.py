import osmnx as ox
import numpy as np
from shapely.geometry import Point, Polygon

def fetch_pois(graph, place_name="Macau, China", selected_pois=None):
    """
    Fetch POIs and snap them to nearest street nodes.
    
    Args:
        graph: NetworkX graph of the area
        place_name: Name of the place to fetch POIs from
        selected_pois: List of POI types to include (e.g., ['bank', 'police', 'school', 'hospital', 'fire_station'])
                      If None, all POIs will be included
    """
    # Define all available POI categories
    all_poi_tags = {
        "shop": ["supermarket", "convenience", "bakery", "butcher", "mall", "department_store"],
        "amenity": ["school", "hospital", "clinic", "pharmacy", "library", "restaurant", "cafe", 
                   "bank", "police", "fire_station", "post_office", "university"],
        "leisure": ["park", "sports_centre", "garden", "playground", "fitness_centre"],
        "office": ["government", "insurance", "company", "lawyer", "financial", "coworking"]
    }
    
    # If selected_pois is provided, filter the tags to only include those POIs
    if selected_pois:
        tags = {}
        for category, poi_types in all_poi_tags.items():
            filtered_types = [poi_type for poi_type in poi_types if poi_type in selected_pois]
            if filtered_types:
                tags[category] = filtered_types
    else:
        # Use all POI tags if no selection is provided
        tags = all_poi_tags
    
    # Initialize POI dictionary with categorization
    pois = {
        # Healthcare
        "healthcare": [],  # hospitals, clinics, pharmacies
        
        # Education
        "education": [],   # schools, libraries, universities
        
        # Shopping
        "shopping": [],    # supermarkets, convenience stores, etc.
        
        # Recreation
        "recreation": [],  # parks, sports centers, etc.
        
        # Services
        "services": [],    # banks, post offices, government, police, fire stations
        
        # Food
        "food": []         # restaurants, cafes
    }
    
    try:
        # Fetch POIs from OSM
        gdf = ox.features_from_place(place_name, tags)
        
        for _, row in gdf.iterrows():
            # Handle both Point and Polygon geometries
            if isinstance(row.geometry, Point):
                x, y = row.geometry.x, row.geometry.y
            elif isinstance(row.geometry, Polygon):
                # Use polygon's centroid if it's an area
                x, y = row.geometry.centroid.x, row.geometry.centroid.y
            else:
                continue  # Skip other geometry types
                
            # Vectorized nearest node search (faster than looping)
            nearest_node = ox.distance.nearest_nodes(graph, x, y)
            
            # Map POI to appropriate category
            if "amenity" in row:
                amenity_type = row["amenity"]
                if amenity_type in ["hospital", "clinic", "pharmacy"]:
                    pois["healthcare"].append((nearest_node, amenity_type))
                elif amenity_type in ["school", "library", "university"]:
                    pois["education"].append((nearest_node, amenity_type))
                elif amenity_type in ["restaurant", "cafe"]:
                    pois["food"].append((nearest_node, amenity_type))
                elif amenity_type in ["bank", "police", "fire_station", "post_office"]:
                    pois["services"].append((nearest_node, amenity_type))
            
            elif "shop" in row:
                shop_type = row["shop"] if isinstance(row["shop"], str) else "shop"
                pois["shopping"].append((nearest_node, shop_type))
            
            elif "leisure" in row:
                leisure_type = row["leisure"] if isinstance(row["leisure"], str) else "leisure"
                pois["recreation"].append((nearest_node, leisure_type))
            
            elif "office" in row:
                office_type = row["office"] if isinstance(row["office"], str) else "office"
                pois["services"].append((nearest_node, office_type))
        
        # Remove duplicates (some POIs might share nearest nodes)
        for category in pois:
            unique_nodes = {}
            for node, poi_type in pois[category]:
                if node not in unique_nodes:
                    unique_nodes[node] = poi_type
            pois[category] = [(node, poi_type) for node, poi_type in unique_nodes.items()]
            
        print(f"Found POIs: {', '.join(f'{len(pois[k])} {k}' for k in pois)}")
        
    except Exception as e:
        print(f"Error fetching POIs: {e}")
        # Create some dummy POIs if real data fetching fails
        # This ensures the simulation can still run
        for category in pois:
            random_nodes = np.random.choice(list(graph.nodes()), size=min(10, len(graph.nodes())), replace=False)
            pois[category] = [(int(node), category) for node in random_nodes]
    
    return pois

# Function to filter POIs to only include specific types
def filter_pois(pois, poi_types=None):
    """
    Filter POIs to only include specific types.
    
    Args:
        pois: Dictionary of POIs by category
        poi_types: List of POI types to include (e.g., ['bank', 'police', 'school', 'hospital', 'fire_station'])
                  If None, all POIs will be included
    
    Returns:
        Filtered POI dictionary
    """
    if poi_types is None:
        return pois
    
    filtered_pois = {category: [] for category in pois}
    
    for category, poi_list in pois.items():
        for poi_entry in poi_list:
            if isinstance(poi_entry, tuple) and len(poi_entry) > 1:
                node, poi_type = poi_entry
                if poi_type in poi_types:
                    filtered_pois[category].append(poi_entry)
    
    return filtered_pois