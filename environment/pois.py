import osmnx as ox
import numpy as np
from shapely.geometry import Point, Polygon
import geopandas as gpd
import pickle
import os

def get_pois_for_category(graph, place_name, tags, poi_type):
    """
    Safely retrieve POIs for a specific category and handle empty results.
    
    Args:
        graph: NetworkX graph of the area
        place_name: Name of the place to fetch POIs from
        tags: Tags to use for OSM query
        poi_type: Type of POI to fetch
        
    Returns:
        List of (node_id, poi_type) tuples
    """
    poi_nodes = []
    try:
        gdf = ox.features_from_place(place_name, tags)
        if len(gdf) > 0:
            print(f"- Found {len(gdf)} {poi_type}")
            
            # Process each POI
            for _, row in gdf.iterrows():
                # Handle both Point and Polygon geometries
                if isinstance(row.geometry, Point):
                    x, y = row.geometry.x, row.geometry.y
                elif isinstance(row.geometry, Polygon):
                    # Use polygon's centroid if it's an area
                    x, y = row.geometry.centroid.x, row.geometry.centroid.y
                else:
                    continue  # Skip other geometry types
                    
                # Find nearest node
                nearest_node = ox.distance.nearest_nodes(graph, x, y)
                poi_nodes.append((nearest_node, poi_type))
        else:
            print(f"- No {poi_type} found")
    except Exception as e:
        print(f"- Error fetching {poi_type}: {e}")
    
    return poi_nodes

def save_pois(pois, filepath):
    """
    Save POIs dictionary to a file using pickle.
    
    Args:
        pois: Dictionary of POIs by category
        filepath: Path where to save the POIs file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(pois, f)
        
        total_pois = sum(len(poi_list) for poi_list in pois.values())
        print(f"POIs saved to {filepath}")
        print(f"Saved {total_pois} POIs across {len(pois)} categories")
    except Exception as e:
        print(f"Error saving POIs: {e}")

def load_pois_from_file(filepath):
    """
    Load POIs dictionary from a saved file.
    
    Args:
        filepath: Path to the saved POIs file
        
    Returns:
        Dictionary of POIs by category or None if loading fails
    """
    try:
        if not os.path.exists(filepath):
            print(f"POIs file not found: {filepath}")
            return None
            
        with open(filepath, 'rb') as f:
            pois = pickle.load(f)
        
        total_pois = sum(len(poi_list) for poi_list in pois.values())
        print(f"POIs loaded from {filepath}")
        print(f"Loaded {total_pois} POIs across {len(pois)} categories")
        return pois
    except Exception as e:
        print(f"Error loading POIs from file: {e}")
        return None

def get_or_fetch_pois(graph, place_name, selected_pois=None, save_path=None, load_path=None):
    """
    Load POIs from file if available, otherwise fetch from OSM and optionally save.
    
    Args:
        graph: NetworkX graph of the area
        place_name: Name of the place to fetch POIs from
        selected_pois: List of POI types to include (optional)
        save_path: Path to save the POIs after fetching (optional)
        load_path: Path to load the POIs from (optional)
        
    Returns:
        Dictionary of POIs by category
    """
    # Try to load from file first if path is provided
    if load_path:
        pois = load_pois_from_file(load_path)
        if pois is not None:
            return pois
        else:
            print("Failed to load POIs from file, fetching from OpenStreetMap...")
    
    # Fetch from OpenStreetMap
    pois = fetch_pois(graph, place_name, selected_pois)
    
    # Save to file if path is provided
    if save_path:
        save_pois(pois, save_path)
    
    return pois

def fetch_pois(graph, place_name="Macau, China", selected_pois=None):
    """
    Fetch POIs and snap them to nearest street nodes.
    
    Args:
        graph: NetworkX graph of the area
        place_name: Name of the place to fetch POIs from
        selected_pois: List of POI types to include (e.g., ['bank', 'police', 'school', 'hospital', 'fire_station'])
                      If None, all POIs will be included
    """
    # Initialize POI dictionary with the requested categories
    pois = {
        "daily_living": [],   # Grocery stores, banks, restaurants, barber shops, post offices
        "healthcare": [],     # Hospitals, clinics, pharmacies
        "education": [],      # Kindergartens, primary schools, secondary schools
        "entertainment": [],  # Parks, libraries, museums, etc.
        "transportation": [], # Bus stops
    }
    
    try:
        # 1. Daily Living POIs
        print("\nFetching daily living points of interest...")
        tags_groceries = {'shop': ['supermarket', 'grocery', 'convenience']}
        tags_banks = {'amenity': 'bank'}
        tags_restaurants = {'amenity': ['restaurant', 'cafe']}
        tags_barber_shops = {'shop': 'hairdresser'}
        tags_post_offices = {'amenity': 'post_office'}
        tags_laundries = {'amenity': 'laundry', 'shop': 'laundry'}
        
        pois["daily_living"].extend(get_pois_for_category(graph, place_name, tags_groceries, 'supermarket'))
        pois["daily_living"].extend(get_pois_for_category(graph, place_name, tags_banks, 'bank'))
        pois["daily_living"].extend(get_pois_for_category(graph, place_name, tags_restaurants, 'restaurant'))
        pois["daily_living"].extend(get_pois_for_category(graph, place_name, tags_barber_shops, 'barber'))
        pois["daily_living"].extend(get_pois_for_category(graph, place_name, tags_post_offices, 'post_office'))
        pois["daily_living"].extend(get_pois_for_category(graph, place_name, tags_laundries, 'laundry'))
        
        # 2. Healthcare POIs
        print("\nFetching healthcare points of interest...")
        tags_hospitals = {'amenity': 'hospital'}
        tags_clinics = {'amenity': 'clinic', 'healthcare': 'clinic'}
        tags_pharmacies = {'amenity': 'pharmacy'}
        tags_health_centers = {'healthcare': 'centre'}
        
        pois["healthcare"].extend(get_pois_for_category(graph, place_name, tags_hospitals, 'hospital'))
        pois["healthcare"].extend(get_pois_for_category(graph, place_name, tags_clinics, 'clinic'))
        pois["healthcare"].extend(get_pois_for_category(graph, place_name, tags_pharmacies, 'pharmacy'))
        pois["healthcare"].extend(get_pois_for_category(graph, place_name, tags_health_centers, 'healthcare'))
        
        # 3. Education POIs
        print("\nFetching education points of interest...")
        tags_kindergartens = {'amenity': 'kindergarten'}
        tags_schools = {'amenity': 'school'}
        
        pois["education"].extend(get_pois_for_category(graph, place_name, tags_kindergartens, 'kindergarten'))
        pois["education"].extend(get_pois_for_category(graph, place_name, tags_schools, 'school'))
        
        # 4. Entertainment POIs
        print("\nFetching entertainment points of interest...")
        tags_parks = {'leisure': 'park'}
        tags_squares = {'place': 'square'}
        tags_libraries = {'amenity': 'library'}
        tags_museums = {'tourism': 'museum'}
        tags_art_galleries = {'tourism': 'gallery'}
        tags_cultural_centers = {'amenity': 'arts_centre'}
        tags_theaters = {'amenity': ['theatre', 'cinema']}
        tags_gyms = {'leisure': ['fitness_centre', 'sports_centre']}
        tags_stadiums = {'leisure': 'stadium'}
        
        pois["entertainment"].extend(get_pois_for_category(graph, place_name, tags_parks, 'park'))
        pois["entertainment"].extend(get_pois_for_category(graph, place_name, tags_squares, 'square'))
        pois["entertainment"].extend(get_pois_for_category(graph, place_name, tags_libraries, 'library'))
        pois["entertainment"].extend(get_pois_for_category(graph, place_name, tags_museums, 'museum'))
        pois["entertainment"].extend(get_pois_for_category(graph, place_name, tags_art_galleries, 'gallery'))
        pois["entertainment"].extend(get_pois_for_category(graph, place_name, tags_cultural_centers, 'cultural_center'))
        pois["entertainment"].extend(get_pois_for_category(graph, place_name, tags_theaters, 'theater'))
        pois["entertainment"].extend(get_pois_for_category(graph, place_name, tags_gyms, 'gym'))
        pois["entertainment"].extend(get_pois_for_category(graph, place_name, tags_stadiums, 'stadium'))
        
        # 5. Transportation POIs
        print("\nFetching transportation points of interest...")
        tags_bus_stops = {'highway': 'bus_stop'}
        
        pois["transportation"].extend(get_pois_for_category(graph, place_name, tags_bus_stops, 'bus_stop'))
        
        # Remove duplicates (some POIs might share nearest nodes)
        for category in pois:
            unique_nodes = {}
            for node, poi_type in pois[category]:
                if node not in unique_nodes:
                    unique_nodes[node] = poi_type
            pois[category] = [(node, poi_type) for node, poi_type in unique_nodes.items()]
            
        print(f"\nFound POIs: {', '.join(f'{len(pois[k])} {k}' for k in pois)}")
        
    except Exception as e:
        print(f"Error fetching POIs: {e}")
        # Create some dummy POIs if real data fetching fails
        # This ensures the simulation can still run
        pois = create_dummy_pois(graph)
    
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
    
    total_filtered = sum(len(poi_list) for poi_list in filtered_pois.values())
    total_original = sum(len(poi_list) for poi_list in pois.values())
    print(f"Filtered POIs: {total_filtered} POIs (from {total_original} total)")
    return filtered_pois

def create_dummy_pois(graph, num_per_category=5):
    """
    Create dummy POIs for testing, with examples for each category.
    
    Args:
        graph: NetworkX graph of the area
        num_per_category: Number of POIs to create per category
        
    Returns:
        Dictionary of POIs by category
    """
    print("Creating dummy POIs for testing...")
    
    # Initialize POI dictionary with the requested categories
    pois = {
        "daily_living": [],   # Grocery stores, banks, restaurants, barber shops, post offices
        "healthcare": [],     # Hospitals, clinics, pharmacies
        "education": [],      # Kindergartens, primary schools, secondary schools
        "entertainment": [],  # Parks, libraries, museums, etc.
        "transportation": [], # Bus stops
    }
    
    # Sample POI types for each category
    category_examples = {
        "daily_living": ['supermarket', 'bank', 'restaurant', 'cafe', 'post_office'],
        "healthcare": ['hospital', 'clinic', 'pharmacy', 'doctor', 'dentist'],
        "education": ['school', 'kindergarten', 'primary_school', 'secondary_school', 'university'],
        "entertainment": ['park', 'library', 'museum', 'theater', 'gym'],
        "transportation": ['bus_stop', 'bus_station', 'station', 'platform', 'stop_position'],
    }
    
    # Get random nodes from the graph
    all_nodes = list(graph.nodes())
    num_nodes_needed = num_per_category * len(pois)
    
    if len(all_nodes) < num_nodes_needed:
        # If not enough nodes, allow reuse
        selected_nodes = [all_nodes[i % len(all_nodes)] for i in range(num_nodes_needed)]
    else:
        # If enough nodes, select without replacement
        selected_nodes = np.random.choice(all_nodes, size=num_nodes_needed, replace=False)
    
    node_index = 0
    
    # Create POIs for each category
    for category, example_types in category_examples.items():
        for i in range(num_per_category):
            poi_type = example_types[i % len(example_types)]
            node_id = selected_nodes[node_index]
            node_index += 1
            
            pois[category].append((node_id, poi_type))
            print(f"Added dummy {poi_type} to {category}")
    
    return pois

def get_or_fetch_residential_buildings(place_name="Macau, China", save_path=None, load_path=None):
    """
    Fetch or load residential buildings for a given place.

    This function first checks if a `load_path` is provided. If so, it loads the
    residential buildings from the specified file. Otherwise, it fetches the data
    from OpenStreetMap (OSM) based on a set of predefined tags for residential
    buildings.

    If a `save_path` is provided, the fetched data will be saved to that path
    for faster access in future runs.

    Args:
        place_name (str): The name of the city or area to fetch buildings for.
        save_path (str, optional): Path to save the fetched buildings.
        load_path (str, optional): Path to load pre-saved buildings from.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the residential buildings.
    """
    if load_path and os.path.exists(load_path):
        print(f"Loading residential buildings from: {load_path}")
        try:
            with open(load_path, 'rb') as f:
                buildings = pickle.load(f)
            print(f"Successfully loaded {len(buildings)} residential buildings from file.")
            return buildings
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading residential buildings from {load_path}: {e}")
            print("Will try fetching from OSM instead.")

    print(f"Fetching residential buildings for {place_name} from OSM...")
    
    # Tags to identify residential buildings in OSM
    tags = {
        "building": ["apartments", "house", "residential", "detached", "semidetached_house", "terrace", "dormitory"]
    }
    
    try:
        buildings = ox.features_from_place(place_name, tags)
        print(f"Successfully fetched {len(buildings)} residential buildings.")

        if save_path:
            print(f"Saving residential buildings to: {save_path}")
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(buildings, f)
        
        return buildings

    except Exception as e:
        print(f"Could not fetch residential buildings from OSM: {e}")
        return gpd.GeoDataFrame()

def filter_pois_by_parishes(pois, graph, parishes_gdf, selected_parishes):
    """
    Filter POIs to only include those within selected parishes.
    """
    if not selected_parishes or parishes_gdf is None:
        return pois
    
    # Get all nodes in the filtered graph
    graph_nodes = set(graph.nodes())
    
    # Filter POIs to only include those in the filtered graph
    filtered_pois = {}
    
    for category, poi_list in pois.items():
        filtered_poi_list = []
        
        for poi_data in poi_list:
            if isinstance(poi_data, tuple):
                node_id, _ = poi_data
            else:
                node_id = poi_data
            
            # Only keep POIs that are in the filtered graph
            if node_id in graph_nodes:
                filtered_poi_list.append(poi_data)
        
        filtered_pois[category] = filtered_poi_list
    
    total_filtered = sum(len(poi_list) for poi_list in filtered_pois.values())
    total_original = sum(len(poi_list) for poi_list in pois.values())
    print(f"Filtered POIs: {total_filtered} POIs (from {total_original} total)")
    return filtered_pois