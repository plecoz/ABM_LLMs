import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.patches import FancyArrowPatch
import numpy as np
import geopandas as gpd
import json
import os
from config.poi_config import get_active_poi_config

from environment.fifteenminutescity.city_network import load_city_network, get_or_load_city_network
from environment.fifteenminutescity.pois import fetch_pois, filter_pois, get_or_fetch_pois, get_or_fetch_environment_data
from simulation.fifteenminutescity.fifteenminutescity_model import FifteenMinuteCity
from visualization import SimulationAnimator
import sys
import re
import unicodedata

# Shapefile path for Macau parishes
CITY_PARISHES_PATHS = {
    "Macau, China": "./data/macau_shapefiles/macau_new_districts.gpkg",
    "Barcelona, Spain": "./data/barcelona_shapefiles/barcelona_districts_clean.gpkg",
    "Hong Kong, China": "./data/hongkong_shapefiles/hongkong_districts.gpkg"
}

# Macau parish population proportions (based on real demographics)
MACAU_PARISH_PROPORTIONS = {
    "Santo Antnio": 0.203,      # 20.3%
    "So Lzaro": 0.05,          # 5%
    "So Loureno": 0.082,       # 8.2%
    "S": 0.083,                 # 8.3%
    "Nossa Senhora de Ftima": 0.368,  # 36.8%
    "Taipa": 0.160,            # 16.0% 
    "Coloane": 0.054,          # 5.4% (formerly So Francisco Xavier)
}

    #--parishes "S" "Nossa Senhora de Ftima" "So Lzaro" "Santo Antnio" "So Loureno" for the old town of macau
    #--parishes "Taipa" "Coloane" for the new city of macau

def get_parishes_path(city):
    """
    Get the parishes shapefile path for a given city.
    
    Args:
        city: Name of the city (e.g., 'Macau, China')
        
    Returns:
        Path to the parishes shapefile or None if not available
    """
    return CITY_PARISHES_PATHS.get(city)

def load_parishes(shapefile_path=None):
    """
    Load parishes from shapefile.
    
    Args:
        shapefile_path: Path to the shapefile containing parish data
        
    Returns:
        GeoDataFrame with parishes data or None if file not found
    """
    if shapefile_path is None:
        print("No parishes shapefile path provided. Simulation will run without parish visualization.")
        return None
        
    try:
        districts = gpd.read_file(shapefile_path)
        print(f"Successfully loaded parishes data!")
        print(f"Number of parishes/districts: {len(districts)}")
        return districts
    except Exception as e:
        print(f"Warning: Could not load parishes data: {e}")
        print("Simulation will run without parish visualization.")
        return None

def load_parish_demographics(demographics_path=None):
    """
    Load parish-specific demographic data from a JSON file.
    
    Args:
        demographics_path: Path to the JSON file with parish-specific demographics
        
    Returns:
        Dictionary of parish demographics or empty dict if file not found
    """
    try:
        if demographics_path and os.path.exists(demographics_path):
            with open(demographics_path, 'r') as f:
                parish_demographics = json.load(f)
            print(f"Successfully loaded parish-specific demographics!")
            print(f"Parishes with custom demographics: {', '.join(parish_demographics.keys())}")
            return parish_demographics
        else:
            print("No parish-specific demographics file provided or file not found.")
            return {}
    except Exception as e:
        print(f"Warning: Could not load parish demographics: {e}")
        return {}

def clean_parish_name(name):
    """
    Remove Chinese characters and accents from parish name for easier matching.
    
    Args:
        name: Original parish name (may contain Chinese characters and accents)
        
    Returns:
        Cleaned parish name with only Latin characters (no accents)
    """
    if not name:
        return name
    
    # Remove Chinese characters (keep only Latin characters, numbers, spaces, and basic punctuation)
    cleaned = re.sub(r'[^\w\s\-\.\(\)]', '', name, flags=re.ASCII)
    
    # Remove accents from letters using Unicode normalization
    # NFD decomposes characters into base + combining marks, then we filter out combining marks
    normalized = unicodedata.normalize('NFD', cleaned)
    without_accents = ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')
    
    # Clean up extra spaces
    final_cleaned = ' '.join(without_accents.split())
    
    return final_cleaned.strip()

def print_available_parishes(parishes_gdf):
    """
    Print all available parish names in both original and cleaned formats.
    
    Args:
        parishes_gdf: GeoDataFrame with parish data
    """
    if parishes_gdf is None:
        print("No parishes data available.")
        return
    
    print("\nAvailable Parishes:")
    print("=" * 50)
    print(f"{'Original Name':<30} | {'Cleaned Name'}")
    print("-" * 50)
    
    for idx, parish in parishes_gdf.iterrows():
        original = parish['name']
        cleaned = clean_parish_name(original)
        print(f"{original:<30} | {cleaned}")
    
    print("=" * 50)
    print(f"Total parishes: {len(parishes_gdf)}")
    print("\nTo use in simulation, copy the 'Cleaned Name' values:")
    cleaned_names = [clean_parish_name(parish['name']) for _, parish in parishes_gdf.iterrows()]
    print(f"--parishes {' '.join([f'\"{name}\"' for name in cleaned_names if name])}")
    print()

def filter_graph_by_parishes(graph, parishes_gdf, selected_parishes):
    """
    Filter the graph to only include nodes within selected parishes.
    """
    if not selected_parishes or parishes_gdf is None:
        return graph
    
    # Clean the selected parish names for matching
    cleaned_selected = [clean_parish_name(name) for name in selected_parishes]
    
    # Create a mapping of cleaned names to original geometries
    parish_geometries = []
    matched_parishes = []
    
    for idx, parish in parishes_gdf.iterrows():
        cleaned_name = clean_parish_name(parish['name'])
        if cleaned_name in cleaned_selected:
            parish_geometries.append(parish['geometry'])
            matched_parishes.append(f"{parish['name']} -> {cleaned_name}")
    
    if not parish_geometries:
        print(f"Warning: No parishes matched the selected names: {selected_parishes}")
        print("Available parishes:")
        for _, parish in parishes_gdf.iterrows():
            print(f"  - Original: '{parish['name']}' -> Cleaned: '{clean_parish_name(parish['name'])}'")
        return graph
    
    print(f"Matched parishes: {matched_parishes}")
    
    # Find nodes within selected parishes
    nodes_to_keep = []
    for node_id, node_attrs in graph.nodes(data=True):
        if 'x' in node_attrs and 'y' in node_attrs:
            from shapely.geometry import Point
            point = Point(node_attrs['x'], node_attrs['y'])
            
            # Check if point is in any selected parish
            for geometry in parish_geometries:
                if geometry.contains(point):
                    nodes_to_keep.append(node_id)
                    break
    
    if not nodes_to_keep:
        print("Warning: No nodes found in selected parishes. Check parish names or coordinate systems.")
        return graph
    
    # Create subgraph with only nodes in selected parishes
    filtered_graph = graph.subgraph(nodes_to_keep).copy()
    print(f"Filtered graph: {len(nodes_to_keep)} nodes (from {len(graph.nodes())} total)")
    
    return filtered_graph

def filter_pois_by_parishes(pois, graph, parishes_gdf, selected_parishes):
    """
    Filter POIs to only include those within selected parishes.
    """
    if not selected_parishes or parishes_gdf is None:
        return pois
    
    # Get all nodes in the filtered graph
    graph_nodes = set(graph.nodes())
    
    # Filter POIs to only include those in the filtered graph
    filtered_pois = {}
    total_original = 0
    total_filtered = 0
    
    for category, poi_list in pois.items():
        filtered_poi_list = []
        total_original += len(poi_list)
        
        for poi_data in poi_list:
            if isinstance(poi_data, tuple):
                node_id, poi_type = poi_data
            else:
                node_id = poi_data
            
            # Only keep POIs that are in the filtered graph
            if node_id in graph_nodes:
                filtered_poi_list.append(poi_data)
        
        filtered_pois[category] = filtered_poi_list
        total_filtered += len(filtered_poi_list)
    
    print(f"Filtered POIs: {total_filtered} POIs (from {total_original} total)")
    return filtered_pois

def calculate_proportional_distribution(selected_parishes, total_residents, random_distribution=False):
    """
    Calculate the number of residents to place in each selected parish.
    
    Args:
        selected_parishes: List of parish names to include
        total_residents: Total number of residents to distribute
        random_distribution: If True, distribute randomly; if False, use proportional distribution
        
    Returns:
        Dictionary mapping parish names to number of residents
    """
    if random_distribution or not selected_parishes:
        # Random distribution - equal number in each parish
        if selected_parishes:
            residents_per_parish = total_residents // len(selected_parishes)
            remainder = total_residents % len(selected_parishes)
            
            distribution = {}
            for i, parish in enumerate(selected_parishes):
                distribution[parish] = residents_per_parish + (1 if i < remainder else 0)
            return distribution
        else:
            return {}
    
    # Proportional distribution based on real demographics
    # Clean parish names for matching
    cleaned_selected = [clean_parish_name(name) for name in selected_parishes]
    
    # Get proportions for selected parishes
    selected_proportions = {}
    total_proportion = 0
    
    for parish in cleaned_selected:
        if parish in MACAU_PARISH_PROPORTIONS:
            selected_proportions[parish] = MACAU_PARISH_PROPORTIONS[parish]
            total_proportion += MACAU_PARISH_PROPORTIONS[parish]
    
    if total_proportion == 0:
        print(f"Warning: No matching parishes found in proportions. Available: {list(MACAU_PARISH_PROPORTIONS.keys())}")
        # Fall back to equal distribution
        residents_per_parish = total_residents // len(selected_parishes)
        remainder = total_residents % len(selected_parishes)
        
        distribution = {}
        for i, parish in enumerate(selected_parishes):
            distribution[parish] = residents_per_parish + (1 if i < remainder else 0)
        return distribution
    
    # Normalize proportions to sum to 1 for selected parishes
    normalized_proportions = {parish: prop / total_proportion 
                            for parish, prop in selected_proportions.items()}
    
    # Calculate number of residents per parish
    distribution = {}
    assigned_residents = 0
    
    # Assign residents based on proportions
    for parish, proportion in normalized_proportions.items():
        num_residents = round(total_residents * proportion)
        distribution[parish] = num_residents
        assigned_residents += num_residents
    
    # Handle rounding differences
    difference = total_residents - assigned_residents
    if difference != 0:
        # Add/subtract residents from the parish with the largest proportion
        largest_parish = max(normalized_proportions.keys(), 
                           key=lambda x: normalized_proportions[x])
        distribution[largest_parish] += difference
    
    # Print distribution summary
    print("\nResident Distribution by Parish:")
    print("=" * 40)
    for parish, count in distribution.items():
        percentage = (count / total_residents) * 100
        print(f"{parish:<25}: {count:>3} residents ({percentage:>5.1f}%)")
    print("=" * 40)
    print(f"Total: {sum(distribution.values())} residents")
    
    return distribution

def run_simulation(num_residents, steps, selected_pois=None, parishes_path=None, parish_demographics_path=None, selected_parishes=None, list_parishes=False, random_distribution=False, needs_selection='random', movement_behavior='need-based', save_network=None, load_network=None, save_pois=None, load_pois=None, save_json_report=None, city='Macau, China', save_environment=None, load_environment=None, seed=42, threshold=15, no_visualization=False):
    """
    Run the 15-minute city simulation.
    
    Args:
        save_network: Path to save the network after loading from OSM
        load_network: Path to load the network from (instead of OSM)
        save_pois: Path to save the POIs after fetching from OSM
        load_pois: Path to load the POIs from (instead of OSM)
        save_environment: Path to save the environment data (buildings, water, cliffs) after fetching from OSM
        load_environment: Path to load the environment data from (instead of OSM)
        save_json_report: Path to save the detailed JSON report (optional)
        city: Name of the city for the simulation (default: 'Macau, China')
        threshold: Time threshold in minutes for accessibility (default: 15)
        no_visualization: If True, run simulation without visualization for faster execution
    """
    # Get parishes path based on city if not explicitly provided
    if parishes_path is None:
        parishes_path = get_parishes_path(city)
        if parishes_path is None:
            print(f"No parishes data available for {city}. Simulation will run without parish visualization.")
    
    # If user just wants to list parishes, do that and exit early
    if list_parishes:
        parishes_gdf = load_parishes(parishes_path)
        print_available_parishes(parishes_gdf)
        return
    
    print(f"Loading {city}'s street network...")
    
    # Use the new save/load functionality for the network
    graph = get_or_load_city_network(
        place_name=city,
        mode="walk",
        save_path=save_network,
        load_path=load_network
    )
    
    # Load parishes data
    parishes_gdf = load_parishes(parishes_path)
    
    # Filter graph by selected parishes if specified
    if selected_parishes and parishes_gdf is not None:
        print(f"Filtering simulation to parishes: {', '.join(selected_parishes)}")
        graph = filter_graph_by_parishes(graph, parishes_gdf, selected_parishes)
        
        # Also filter parishes_gdf to only selected parishes
        cleaned_selected = [clean_parish_name(name) for name in selected_parishes]
        parishes_gdf = parishes_gdf[parishes_gdf['name'].apply(clean_parish_name).isin(cleaned_selected)]
    
    # Calculate proportional distribution
    parish_distribution = None
    if parishes_gdf is not None and not random_distribution:
        # If specific parishes are selected, use only those
        if selected_parishes:
            parish_distribution = calculate_proportional_distribution(
                selected_parishes, num_residents, random_distribution
            )
        else:
            # If no specific parishes selected, use all available parishes with proportional distribution
            all_parish_names = [clean_parish_name(parish['name']) for _, parish in parishes_gdf.iterrows()]
            # Filter out empty names
            all_parish_names = [name for name in all_parish_names if name]
            if all_parish_names:
                print("Using proportional distribution across all Macau parishes")
                parish_distribution = calculate_proportional_distribution(
                    all_parish_names, num_residents, random_distribution
                )
    
    # Load parish-specific demographics
    parish_demographics = load_parish_demographics(parish_demographics_path)
    
    # Get POI configuration from config file if not explicitly provided
    if selected_pois is None:
        selected_pois = get_active_poi_config()
    
    if selected_pois:
        print(f"Using selected POI types: {', '.join(selected_pois)}")
    else:
        print("Using all available POI types")
    
    # Use the new save/load functionality for POIs
    pois = get_or_fetch_pois(
        graph=graph,
        place_name=city,
        selected_pois=selected_pois,
        save_path=save_pois,
        load_path=load_pois
    )
    
    # Fetch or load environment data (buildings, water bodies, cliffs)
    environment_data = get_or_fetch_environment_data(
        place_name=city,
        save_path=save_environment,
        load_path=load_environment
    )
    
    # Extract components from environment data
    residential_buildings = environment_data.get('residential_buildings', gpd.GeoDataFrame())
    water_bodies = environment_data.get('water_bodies', gpd.GeoDataFrame())
    cliffs = environment_data.get('cliffs', gpd.GeoDataFrame())
    forests = environment_data.get('forests', gpd.GeoDataFrame())
    
    if not residential_buildings.empty:
        print(f"Found {len(residential_buildings)} residential buildings.")
    else:
        print("No residential buildings found or loaded.")
        
    if not water_bodies.empty:
        print(f"Found {len(water_bodies)} water bodies.")
    else:
        print("No water bodies found or loaded.")
        
    if not cliffs.empty:
        print(f"Found {len(cliffs)} cliffs and barriers.")
    else:
        print("No cliffs and barriers found or loaded.")
        
    if not forests.empty:
        print(f"Found {len(forests)} forests and green areas.")
    else:
        print("No forests and green areas found or loaded.")
    
    # Filter POIs by selected parishes
    if selected_parishes and parishes_gdf is not None:
        pois = filter_pois_by_parishes(pois, graph, parishes_gdf, selected_parishes)
    
    print(f"Spawning {num_residents} residents...")
    model = FifteenMinuteCity(
        graph=graph,
        pois=pois,
        num_residents=num_residents,
        parishes_gdf=parishes_gdf,
        parish_demographics=parish_demographics,
        parish_distribution=parish_distribution,
        random_distribution=random_distribution,
        needs_selection=needs_selection,
        movement_behavior=movement_behavior,
        city=city,
        residential_buildings=residential_buildings,
        seed=seed,
        threshold=threshold
    )
    
    print("Starting simulation...")
    
    if no_visualization:
        # Run simulation without visualization (headless mode) - much faster
        print("Running in headless mode (no visualization)...")
        print(f"Simulating {steps} time steps...")
        
        # Run the simulation steps directly
        for step in range(steps):
            model.step()
            
            # Print progress every 60 steps (1 hour in simulation time)
            if step % 60 == 0 or step == steps - 1:
                time_info = model.get_current_time()
                print(f"Step {step + 1}/{steps} - {time_info['time_string']}")
        
        print("Simulation completed!")
        
        # Print final statistics
        if hasattr(model, 'output_controller'):
            model.output_controller.print_travel_summary()
    else:
        # Run simulation with visualization
        print("Running with visualization...")
        
        # Set up interactive mode
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Initialize animator with parishes data and environment components
        animator = SimulationAnimator(
            model, 
            graph, 
            ax=ax, 
            parishes_gdf=parishes_gdf,
            residential_buildings=residential_buildings,
            water_bodies=water_bodies,
            cliffs=cliffs,
            forests=forests
        )
        
        # Configure animator to use specific styling for selected POIs
        if selected_pois:
            animator.use_specific_poi_styling = True
        
        animator.initialize()  # Draw initial state
        
        # Start the animation loop
        animator.start_animation(steps, interval=50)  # 50ms between frames
        
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep window open at end
    
    # Save JSON report if requested (works for both modes)
    if save_json_report:
        print(f"\nSaving detailed JSON report to: {save_json_report}")
        model.output_controller.save_detailed_report(save_json_report)

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run the 15-minute city simulation with Macau parishes')
    
    # Add seed argument at the top for visibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible results (default: 42)')
    
    parser.add_argument('--residents', type=int, default=10, help='Number of resident agents')
    parser.add_argument('--steps', type=int, default=5, help='Number of simulation steps (1 step = 1 minute, default: 480 = 8 hours)')
    parser.add_argument('--threshold', type=int, default=100, help='Time threshold in minutes for accessibility (default: 15 for 15-minute city, use 10 for 10-minute city, etc.)')
    parser.add_argument('--parishes-path', type=str, help='Path to parishes/districts shapefile')
    parser.add_argument('--parish-demographics', type=str, help='Path to parish-specific demographics JSON file', default=r"C:\Users\pierr\UNU_macau\ABM_LLMs\data\demographics_macau\parish_demographic.json")
    
    parser.add_argument('--parishes', nargs='+', help='List of parish names to include in simulation (e.g., --parishes "Parish A" "Parish B")')
    #--parishes "S" "Nossa Senhora de Ftima" "So Lzaro" "Santo Antnio" "So Loureno" for the old town of macau
    #--parishes "Taipa" "Coloane" for the new city of macau
    parser.add_argument('--list-parishes', action='store_true', help='List all available parish names and exit')
    parser.add_argument('--random-distribution', action='store_true', help='Distribute residents randomly across parishes instead of using proportional distribution (default: False)')
    parser.add_argument('--needs-selection', type=str, choices=['random', 'maslow', 'capability', 'llms'], default='random', help='Method for generating resident needs (default: random)')
    parser.add_argument('--movement-behavior', type=str, choices=['need-based', 'random', 'llms'], default='random', help='Agent movement behavior: need-based (agents go to POIs to satisfy needs) or random (agents move randomly) or llms (agents move to POIs based on LLM instructions) (default: need-based)')
    
    # Save/Load arguments for faster testing
    parser.add_argument('--save-network', type=str, help='Path to save the city network after loading from OSM (e.g., data/macau_network.pkl)')
    #python main.py --save-network data/macau_network.pkl --save-pois data/macau_pois.pkl
    parser.add_argument('--load-network', type=str, help='Path to load the city network from file instead of OSM (e.g., data/macau_network.pkl)')
    #python main.py --load-network data/macau_shapefiles/macau_network.pkl --load-pois data/macau_shapefiles/macau_pois.pkl
    #python main.py --load-network data/barcelona_shapefiles/barcelona_network.pkl --load-pois data/barcelona_shapefiles/barcelona_pois.pkl
    parser.add_argument('--save-pois', type=str, help='Path to save the POIs after fetching from OSM (e.g., data/macau_pois.pkl)')
    parser.add_argument('--load-pois', type=str, help='Path to load the POIs from file instead of OSM (e.g., data/macau_pois.pkl)')
    parser.add_argument('--save-environment', type=str, help='Path to save the environment data (buildings, water, cliffs) after fetching from OSM (e.g., data/macau_environment.pkl)')
    parser.add_argument('--load-environment', type=str, help='Path to load the environment data from file instead of OSM (e.g., data/macau_environment.pkl)')
    
    # JSON report argument
    parser.add_argument('--save-json-report', type=str, help='Path to save the detailed JSON simulation report (e.g., reports/simulation_report.json)')
    
    # City argument
    parser.add_argument('--city', type=str, default='Macau, China', help='City name for the simulation (default: Macau, China)')
    
    # No visualization argument
    parser.add_argument('--no-visualization', action='store_true', help='Run simulation without visualization for faster execution')
    
    args = parser.parse_args()
    #Good simulations :
    #python main.py --load-network data/barcelona_shapefiles/barcelona_network.pkl --load-pois data/barcelona_shapefiles/barcelona_pois.pkl --parishes "Ciutat Vella"
    
    # Use configuration from config file
    selected_pois = get_active_poi_config()
    
    run_simulation(
        num_residents=args.residents, 
        steps=args.steps, 
        selected_pois=selected_pois,
        parishes_path=args.parishes_path,
        parish_demographics_path=args.parish_demographics,
        selected_parishes=args.parishes,
        list_parishes=args.list_parishes,
        random_distribution=args.random_distribution,
        needs_selection=args.needs_selection,
        movement_behavior=args.movement_behavior,
        save_network=args.save_network,
        load_network=args.load_network,
        save_pois=args.save_pois,
        load_pois=args.load_pois,
        save_json_report=args.save_json_report,
        city=args.city,
        save_environment=args.save_environment,
        load_environment=args.load_environment,
        seed=args.seed,
        threshold=args.threshold,
        no_visualization=args.no_visualization
    )