import osmnx as ox
import networkx as nx
import geopandas as gpd
import pickle
import os

def load_city_network(place_name="Macau, China", mode="walk"):  
    """
    Load city's walkable network using OSMnx with slope and speed limit data.
    
    Args:
        place_name: Name of the place to load network for
        mode: Network type (walk, drive, bike, etc.)
        
    Returns:
        NetworkX graph representing the street network
    """
    print(f"Loading {mode} network for {place_name}...")
    
    # Configure OSMnx settings
    ox.settings.bidirectional_network_types = ['walk']
    ox.settings.use_cache = True
    
    # Add minimal useful tags for slopes and speed limits
    ox.settings.useful_tags_way = ['maxspeed', 'incline']
    
    graph = ox.graph_from_place(
        place_name,
        network_type=mode,
        simplify=True,  # Clean topological artifacts
        retain_all=True,  # Keep all edges
    )
    
    # Add elevation data for slope calculation
    try:
        print("Fetching elevation data for slope calculation...")
        graph = ox.elevation.add_node_elevations_raster(graph, raster_path=None, cpus=1)
        graph = ox.elevation.add_edge_grades(graph)
        print("✓ Added elevation and slope data")
    except Exception as e:
        print(f"⚠ Could not add elevation data: {e}")
        print("  Will use OSM incline tags only")
    
    # Verify conversion and check data coverage
    print(f"Graph loaded with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    # Check percentage of edges with slope and speed data
    total_edges = len(graph.edges())
    edges_with_slopes = 0
    edges_with_speed_limits = 0
    
    for u, v, data in graph.edges(data=True):
        # Check for slope data (either calculated grade or OSM incline)
        if data.get('grade') is not None or data.get('incline') is not None:
            edges_with_slopes += 1
        
        # Check for speed limit data
        if data.get('maxspeed') is not None:
            edges_with_speed_limits += 1
    
    slope_percentage = (edges_with_slopes / total_edges) * 100 if total_edges > 0 else 0
    speed_percentage = (edges_with_speed_limits / total_edges) * 100 if total_edges > 0 else 0
    
    print(f"Data coverage:")
    print(f"  - Edges with slope data: {edges_with_slopes}/{total_edges} ({slope_percentage:.1f}%)")
    print(f"  - Edges with speed limits: {edges_with_speed_limits}/{total_edges} ({speed_percentage:.1f}%)")
    
    return graph

def save_city_network(graph, filepath):
    """
    Save a NetworkX graph to a file using pickle.
    
    Args:
        graph: NetworkX graph to save
        filepath: Path where to save the graph file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        print(f"City network saved to {filepath}")
        print(f"Saved graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    except Exception as e:
        print(f"Error saving city network: {e}")

def load_city_network_from_file(filepath):
    """
    Load a NetworkX graph from a saved file.
    
    Args:
        filepath: Path to the saved graph file
        
    Returns:
        NetworkX graph or None if loading fails
    """
    try:
        if not os.path.exists(filepath):
            print(f"Network file not found: {filepath}")
            return None
            
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
        print(f"City network loaded from {filepath}")
        print(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        return graph
    except Exception as e:
        print(f"Error loading city network from file: {e}")
        return None

def get_or_load_city_network(place_name, mode="walk", save_path=None, load_path=None):
    """
    Load city network from file if available, otherwise fetch from OSM and optionally save.
    
    Args:
        place_name: Name of the place to load network for
        mode: Network type (walk, drive, bike, etc.)
        save_path: Path to save the network after fetching (optional)
        load_path: Path to load the network from (optional)
        
    Returns:
        NetworkX graph representing the street network
    """
    # Try to load from file first if path is provided
    if load_path:
        graph = load_city_network_from_file(load_path)
        if graph is not None:
            return graph
        else:
            print("Failed to load from file, fetching from OpenStreetMap...")
    
    # Fetch from OpenStreetMap
    graph = load_city_network(place_name, mode)
    
    # Save to file if path is provided
    if save_path:
        save_city_network(graph, save_path)
    
    return graph

def project_graph_to_crs(graph, crs="EPSG:4326"):
    """
    Project graph to a specific coordinate reference system.
    
    Args:
        graph: NetworkX graph to project
        crs: Target coordinate reference system
        
    Returns:
        Projected NetworkX graph
    """
    return ox.project_graph(graph, to_crs=crs)

def clip_graph_to_boundaries(graph, boundary_gdf):
    """
    Clip a street network graph to a boundary polygon.
    
    Args:
        graph: NetworkX graph to clip
        boundary_gdf: GeoDataFrame containing boundary polygon(s)
        
    Returns:
        Clipped NetworkX graph
    """
    return ox.truncate.truncate_graph_polygon(graph, boundary_gdf.unary_union, retain_all=True)

def get_slope_from_edge(edge_data):
    """
    Get slope percentage from edge data.
    
    Args:
        edge_data: Edge attributes dictionary
        
    Returns:
        Slope percentage or None
    """
    # Try elevation-based grade first
    grade = edge_data.get('grade')
    if grade is not None:
        return round(grade, 1)
    
    # Try OSM incline tag
    incline = edge_data.get('incline')
    if incline:
        incline_str = str(incline).lower().strip()
        
        # Handle percentage format
        if '%' in incline_str:
            import re
            numbers = re.findall(r'-?\d+\.?\d*', incline_str)
            if numbers:
                return round(float(numbers[0]), 1)
        
        # Handle simple up/down
        if incline_str == 'up':
            return 5.0
        elif incline_str == 'down':
            return -5.0
    
    return None

def get_speed_limit_from_edge(edge_data):
    """
    Get speed limit from edge data.
    
    Args:
        edge_data: Edge attributes dictionary
        
    Returns:
        Speed limit in km/h or None
    """
    maxspeed = edge_data.get('maxspeed')
    if not maxspeed:
        return None
    
    # Handle list/tuple
    if isinstance(maxspeed, (list, tuple)):
        maxspeed = maxspeed[0] if maxspeed else None
    
    if not maxspeed:
        return None
    
    speed_str = str(maxspeed).lower().strip()
    
    # Extract number
    import re
    numbers = re.findall(r'\d+', speed_str)
    if not numbers:
        return None
    
    speed = int(numbers[0])
    
    # Convert mph to km/h
    if 'mph' in speed_str:
        speed = int(speed * 1.60934)
    
    return speed if 0 <= speed <= 200 else None