import osmnx as ox
import networkx as nx

def load_city_network(place_name="Macau, China", mode="walk"):  
    """Load Macau's walkable network using OSMnx."""
    ox.settings.bidirectional_network_types = ['walk']
    graph = ox.graph_from_place(
        place_name,
        network_type=mode,
        simplify=True,  # Clean topological artifacts
        retain_all=True  # Keep all edges
    )
    
    # Verify conversion
    print(f"Graph type: {type(graph)}")
    print(f"Edges: {len(graph.edges())} (should be >0)")
    
    return graph