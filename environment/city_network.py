import osmnx as ox
import networkx as nx
import geopandas as gpd

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