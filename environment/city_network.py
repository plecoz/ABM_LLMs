import osmnx as ox
import networkx as nx

def load_city_network(place_name="Paris, France", mode="walk"):
    """Load and simplify a city's walk/bike network using OSMnx."""
    graph = ox.graph_from_place(place_name, network_type=mode)
    graph = ox.utils_graph.get_undirected(graph)
    return graph