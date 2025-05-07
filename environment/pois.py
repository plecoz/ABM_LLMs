import osmnx as ox

def fetch_pois(graph, place_name="Paris, France"):
    """Fetch Points of Interest (POIs) from OpenStreetMap."""
    tags = {
        "shop": True,
        "amenity": ["school", "hospital"],
        "leisure": ["park"],
    }
    pois = {k: [] for k in ["shop", "school", "park"]}
    
    gdf = ox.geometries_from_place(place_name, tags)
    for _, row in gdf.iterrows():
        nearest_node = ox.distance.nearest_nodes(graph, row.geometry.x, row.geometry.y)
        if "shop" in row:
            pois["shop"].append(nearest_node)
        elif "amenity" in row and row["amenity"] == "school":
            pois["school"].append(nearest_node)
        elif "leisure" in row and row["leisure"] == "park":
            pois["park"].append(nearest_node)
    return pois