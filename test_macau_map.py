import osmnx as ox
import matplotlib.pyplot as plt

ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 180  # Increase timeout to 3 minutes

try:
    # Download Macau street network
    print("Downloading 'Macau, China' street network...")
    graph = ox.graph_from_place(
        "Macau, China", 
        network_type="walk",  # Pedestrian paths only
        simplify=True        # Clean topological artifacts
    )
    
    print(f"Success! Loaded:")
    print(f"- Nodes (intersections): {len(graph.nodes())}")
    print(f"- Edges (streets): {len(graph.edges())}")
    print(f"- Total walkable length: {ox.stats.edge_length_total(graph)/1000:.1f} km")
    
    # Get schools and hospitals
    print("\nDownloading points of interest...")
    tags_schools = {'amenity': 'school'}
    tags_hospitals = {'amenity': 'hospital'}
    
    schools = ox.features_from_place('Macau, China', tags_schools)
    hospitals = ox.features_from_place('Macau, China', tags_hospitals)
    
    print(f"- Found {len(schools)} schools")
    print(f"- Found {len(hospitals)} hospitals")
    
    # Create figure with single axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot street network first (background)
    ox.plot_graph(
        graph,
        bgcolor="#f5f5f5",  # Light gray background
        node_size=0,        # Hide nodes
        edge_color="#4682b4",  # Steel blue streets
        edge_linewidth=0.7,
        show=False,
        ax=ax
    )
    
    # Plot schools and hospitals on the same axis
    schools_centroids = schools.geometry.centroid
    hospitals_centroids = hospitals.geometry.centroid
    
    # Create scatter plots with labels for legend
    schools_centroids.plot(ax=ax, color='red', markersize=50, alpha=0.7, label='Schools')
    hospitals_centroids.plot(ax=ax, color='green', markersize=50, alpha=0.7, label='Hospitals')
    
    # Add title and legend
    ax.set_title("Macau Walkable Streets with Schools and Hospitals", fontsize=16, pad=20)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\n❌ Error: {e}")