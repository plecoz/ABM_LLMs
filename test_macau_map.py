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
    tags_banks = {'amenity': 'bank'}
    tags_police = {'amenity': 'police'}
    tags_fire = {'amenity': 'fire_station'}
    
    schools = ox.features_from_place('Macau, China', tags_schools)
    hospitals = ox.features_from_place('Macau, China', tags_hospitals)
    banks = ox.features_from_place('Macau, China', tags_banks)
    police = ox.features_from_place('Macau, China', tags_police)
    fire_stations = ox.features_from_place('Macau, China', tags_fire)
    
    print(f"- Found {len(schools)} schools")
    print(f"- Found {len(hospitals)} hospitals")
    print(f"- Found {len(banks)} banks")
    print(f"- Found {len(police)} police stations")
    print(f"- Found {len(fire_stations)} fire stations")
    
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
    
    # Plot POIs on the same axis
    schools_centroids = schools.geometry.centroid
    hospitals_centroids = hospitals.geometry.centroid
    banks_centroids = banks.geometry.centroid
    police_centroids = police.geometry.centroid
    fire_stations_centroids = fire_stations.geometry.centroid
    
    # Create scatter plots with labels for legend
    schools_centroids.plot(ax=ax, color='red', markersize=50, alpha=0.7, label='Schools')
    hospitals_centroids.plot(ax=ax, color='green', markersize=50, alpha=0.7, label='Hospitals')
    banks_centroids.plot(ax=ax, color='blue', markersize=50, alpha=0.7, label='Banks')
    police_centroids.plot(ax=ax, color='purple', markersize=50, alpha=0.7, label='Police Stations')
    fire_stations_centroids.plot(ax=ax, color='orange', markersize=50, alpha=0.7, label='Fire Stations')
    
    # Add title and legend
    ax.set_title("Macau Walkable Streets with Essential Services", fontsize=16, pad=20)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('macau_essential_services.png', dpi=300)
    plt.show()

except Exception as e:
    print(f"\n❌ Error: {e}")