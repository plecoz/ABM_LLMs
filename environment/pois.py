import osmnx as ox

import osmnx as ox
import numpy as np
from shapely.geometry import Point, Polygon

def fetch_pois(graph, place_name="Macau, China"):
    """Fetch POIs and snap them to nearest street nodes."""
    tags = {
        "shop": True,
        "amenity": ["school", "hospital"],
        "leisure": ["park"]
    }
    pois = {k: [] for k in ["shop", "school", "park"]}
    
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
        
        # Categorize POIs
        if "shop" in row:
            pois["shop"].append(nearest_node)
        elif "amenity" in row and row["amenity"] == "school":
            pois["school"].append(nearest_node)
        elif "leisure" in row and row["leisure"] == "park":
            pois["park"].append(nearest_node)
    
    # Remove duplicates (some POIs might share nearest nodes)
    for category in pois:
        pois[category] = list(set(pois[category]))
        
    return pois