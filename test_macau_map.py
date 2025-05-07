import osmnx as ox
import matplotlib.pyplot as plt

import osmnx as ox
import matplotlib.pyplot as plt

ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 180  # Increase timeout to 3 minutes

try:
  
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
    
    
    print("\nPlotting... (this may take a moment for large areas)")
    fig, ax = plt.subplots(figsize=(12, 10))
    ox.plot_graph(
        graph,
        bgcolor="#f5f5f5",  # Light gray background
        node_size=0,        # Hide nodes
        edge_color="#4682b4",  # Steel blue streets
        edge_linewidth=0.7,
        show=False
    )
    ax.set_title("Macau Walkable Streets (Full Area)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\n❌ Error: {e}")