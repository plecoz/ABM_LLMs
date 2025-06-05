import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox

shapefile_path = "C:/Users/pierr/OneDrive/Documents/Stage Macau/shapefiles/macau_shapefiles/macau_districts.gpkg"

# Read the shapefile
try:
    # Open the shapefile
    districts = gpd.read_file(shapefile_path)
    
    # Print basic information
    print("Successfully loaded shapefile!")
    print(f"Number of districts: {len(districts)}")
    print("\nFirst few rows of data:")
    print(districts.head())
    
    # Print column names
    print("\nAvailable columns:")
    print(districts.columns.tolist())
    
    # Get the walking path graph for Macau
    graph = ox.graph_from_place(
        "Macau, China",
        network_type="walk",
        simplify=True,
        retain_all=True
    )
    
    # Verify conversion
    print(f"\nGraph type: {type(graph)}")
    print(f"Edges: {len(graph.edges())} (should be >0)")
    
    # Plot both the districts and the walking paths
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot districts with colors
    districts.plot(ax=ax, column='name', legend=True, cmap='tab20', edgecolor='black', alpha=0.5)
    
    # Plot the walking path graph
    ox.plot_graph(graph, ax=ax, node_size=0, edge_linewidth=0.5, edge_color='gray', show=False)
    
    # Add labels (adjust the text position if needed)
    for x, y, label in zip(districts.geometry.centroid.x, 
                          districts.geometry.centroid.y, 
                          districts['name']):
        ax.text(x, y, label, fontsize=8, ha='center')
    
    # Add map elements
    plt.title('Macau Districts with Walking Paths')
    ax.set_axis_off()  # Turn off axis
    plt.tight_layout()
    
    # Show the plot
    plt.show()

except FileNotFoundError:
    print(f"Error: Could not find the shapefile at {shapefile_path}")
    print("Please verify:")
    print(f"1. The file exists at: {shapefile_path}")
    print("2. You have read permissions for this file")
except Exception as e:
    print(f"An error occurred: {str(e)}")