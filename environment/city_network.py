import osmnx as ox
import networkx as nx
import geopandas as gpd
import pickle
import os

def load_city_network(place_name="Macau, China", mode="walk"):  
    """
    Load Macau's walkable network using OSMnx.
    
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
    
    graph = ox.graph_from_place(
        place_name,
        network_type=mode,
        simplify=True,  # Clean topological artifacts
        retain_all=True  # Keep all edges
    )
    
    # Verify conversion
    print(f"Graph loaded with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
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