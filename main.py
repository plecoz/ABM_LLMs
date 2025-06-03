from environment.city_network import load_city_network
from environment.pois import fetch_pois, filter_pois, create_dummy_pois
from simulation.model import FifteenMinuteCity
from visualization import SimulationAnimator
import matplotlib.pyplot as plt
import sys
import os
import geopandas as gpd
import json
import re
import unicodedata

# Import POI configuration
try:
    from config.poi_config import get_active_poi_config
except ImportError:
    # Create config directory if it doesn't exist
    if not os.path.exists('config'):
        os.makedirs('config')
    print("POI configuration not found. Using default configuration (all POIs).")
    get_active_poi_config = lambda: None

# Shapefile path for Macau parishes
DEFAULT_PARISHES_PATH = "./data/macau_shapefiles/macau_districts.gpkg"

# Macau parish population proportions (based on real demographics)
MACAU_PARISH_PROPORTIONS = {
    "Santo Antnio": 0.203,      # 20.3%
    "So Lzaro": 0.05,          # 5%
    "So Loureno": 0.082,       # 8.2%
    "S": 0.083,                 # 8.3%
    "Nossa Senhora de Ftima": 0.368,  # 36.8%
    "Nossa Senhora do Carmo": 0.155,   # 15.5%
    "So Francisco Xavier": 0.054,     # 5.4%
    "Zona do Aterro de Cotai": 0.005   # 0.5%
}

    #--parishes "S" "Nossa Senhora de Ftima" "So Lzaro" "Santo Antnio" "So Loureno" for the old town of macau
    #--parishes "So Francisco Xavier" "Nossa Senhora do Carmo" "Zona do Aterro de Cotai" for the new city of macau

def load_parishes(shapefile_path=None):
    """
    Load Macau parishes from shapefile.
    
    Args:
        shapefile_path: Path to the shapefile containing parish data
        
    Returns:
        GeoDataFrame with parishes data or None if file not found
    """
    try:
        path = shapefile_path or DEFAULT_PARISHES_PATH
        districts = gpd.read_file(path)
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

def create_example_parish_demographics(parishes_gdf, output_path='config/parish_demographics.json'):
    """
    Create an example parish demographics file with different income distributions per parish.
    
    Args:
        parishes_gdf: GeoDataFrame with parishes
        output_path: Path to save the example demographics file
        
    Returns:
        Dictionary with the created demographics
    """
    if parishes_gdf is None:
        print("No parishes data available. Cannot create example demographics.")
        return {}
        
    parish_demographics = {}
    
    # Create varying demographics for each parish
    for i, parish in parishes_gdf.iterrows():
        parish_name = parish['name']
        
        # Create different income distributions based on parish index
        # This is just an example - in a real scenario, you'd use actual data
        if i % 3 == 0:  # Wealthy parishes
            income_dist = {"low": 0.1, "medium": 0.3, "high": 0.6}
            income_ranges = {
                "low": (30000, 50000),
                "medium": (50001, 150000),
                "high": (150001, 1000000)
            }
        elif i % 3 == 1:  # Middle-class parishes
            income_dist = {"low": 0.2, "medium": 0.6, "high": 0.2}
            income_ranges = {
                "low": (15000, 40000),
                "medium": (40001, 120000),
                "high": (120001, 500000)
            }
        else:  # Working-class parishes
            income_dist = {"low": 0.6, "medium": 0.3, "high": 0.1}
            income_ranges = {
                "low": (8000, 25000),
                "medium": (25001, 80000),
                "high": (80001, 300000)
            }
            
        # Also vary age distributions
        if i % 2 == 0:  # Younger parishes
            age_dist = {"0-18": 0.3, "19-35": 0.4, "36-65": 0.2, "65+": 0.1}
        else:  # Older parishes
            age_dist = {"0-18": 0.1, "19-35": 0.2, "36-65": 0.4, "65+": 0.3}
            
        parish_demographics[parish_name] = {
            "income_distribution": income_dist,
            "income_ranges": income_ranges,
            "age_distribution": age_dist,
            # Keep the same gender and education distributions as global
            "gender_distribution": {"male": 0.49, "female": 0.49, "other": 0.02},
            "education_distribution": {
                "no_education": 0.1,
                "primary": 0.2,
                "high_school": 0.4,
                "bachelor": 0.2,
                "master": 0.08,
                "phd": 0.02
            }
        }
    
    # Save the example demographics to a file
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(parish_demographics, f, indent=4)
        print(f"Example parish demographics saved to {output_path}")
    except Exception as e:
        print(f"Warning: Could not save example demographics: {e}")
    
    return parish_demographics

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

def run_simulation(num_residents, steps, selected_pois=None, parishes_path=None, parish_demographics_path=None, create_example_demographics=False, use_dummy_pois=False, selected_parishes=None, list_parishes=False, random_distribution=False, needs_selection='random', movement_behavior='need-based'):
    """
    Run the 15-minute city simulation.
    """
    print("Loading Macau's street network...")
    graph = load_city_network("Macau, China")
    
    # Load parishes data
    parishes_gdf = load_parishes(parishes_path)
    
    # If user wants to list parishes, do that and exit
    if list_parishes:
        print_available_parishes(parishes_gdf)
        return
    
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
    
    # Load or create parish-specific demographics
    parish_demographics = {}
    if create_example_demographics and parishes_gdf is not None:
        parish_demographics = create_example_parish_demographics(parishes_gdf)
    else:
        parish_demographics = load_parish_demographics(parish_demographics_path)
    
    # Get POI configuration from config file if not explicitly provided
    if selected_pois is None:
        selected_pois = get_active_poi_config()
    
    if selected_pois:
        print(f"Using selected POI types: {', '.join(selected_pois)}")
    else:
        print("Using all available POI types")
    
    # Fetch POIs - either dummy or from OSM
    if use_dummy_pois:
        print("Using dummy POIs for testing...")
        pois = create_dummy_pois(graph, num_per_category=5)
    else:
        # Fetch POIs with selected types
        pois = fetch_pois(graph, selected_pois=selected_pois)
    
    # Filter POIs by selected parishes
    if selected_parishes and parishes_gdf is not None:
        pois = filter_pois_by_parishes(pois, graph, parishes_gdf, selected_parishes)
    

    
    print(f"Spawning {num_residents} residents...")
    model = FifteenMinuteCity(
        graph, 
        pois, 
        num_residents=num_residents,
        parishes_gdf=parishes_gdf,
        parish_demographics=parish_demographics,
        parish_distribution=parish_distribution,
        random_distribution=random_distribution,
        needs_selection=needs_selection,
        movement_behavior=movement_behavior
    )
    

    print("Starting simulation...")
    # Set up interactive mode
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Initialize animator with parishes data
    animator = SimulationAnimator(model, graph, ax=ax, parishes_gdf=parishes_gdf)
    
    # Configure animator to use specific styling for selected POIs
    if selected_pois:
        animator.use_specific_poi_styling = True
    
    animator.initialize()  # Draw initial state
    
    # Start the animation loop
    animator.start_animation(steps, interval=50)  # 50ms between frames
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep window open at end

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run the 15-minute city simulation with Macau parishes')
    parser.add_argument('--essential-only', action='store_true', help='Only use essential services POIs')
    parser.add_argument('--all-pois', action='store_true', help='Use all available POI types')
    parser.add_argument('--residents', type=int, default=100, help='Number of resident agents')
    parser.add_argument('--steps', type=int, default=480, help='Number of simulation steps (1 step = 1 minute, default: 480 = 8 hours)')
    parser.add_argument('--parishes-path', type=str, help='Path to parishes/districts shapefile')
    parser.add_argument('--parish-demographics', type=str, help='Path to parish-specific demographics JSON file')
    parser.add_argument('--create-example-demographics', action='store_true', help='Create example parish demographics')
    parser.add_argument('--use-dummy-pois', action='store_true', help='Use dummy POIs for testing')
    
    parser.add_argument('--parishes', nargs='+', help='List of parish names to include in simulation (e.g., --parishes "Parish A" "Parish B")')
    #--parishes "S" "Nossa Senhora de Ftima" "So Lzaro" "Santo Antnio" "So Loureno" for the old town of macau
    #--parishes "So Francisco Xavier" "Nossa Senhora do Carmo" "Zona do Aterro de Cotai" for the new city of macau
    parser.add_argument('--list-parishes', action='store_true', help='List all available parish names and exit')
    parser.add_argument('--random-distribution', action='store_true', help='Distribute residents randomly across parishes instead of using proportional distribution (default: False)')
    parser.add_argument('--needs-selection', type=str, choices=['random', 'maslow', 'capability', 'llms'], default='random', help='Method for generating resident needs (default: random)')
    parser.add_argument('--movement-behavior', type=str, choices=['need-based', 'random'], default='random', help='Agent movement behavior: need-based (agents go to POIs to satisfy needs) or random (agents move randomly) (default: need-based)')
    
    args = parser.parse_args()
    
    # Get POI configuration
    if args.essential_only:
        try:
            from config.poi_config import ESSENTIAL_SERVICES_ONLY
            selected_pois = ESSENTIAL_SERVICES_ONLY
        except ImportError:
            print("Essential services configuration not found. Using all POIs.")
            selected_pois = None
    elif args.all_pois:
        selected_pois = None
    else:
        # Use configuration from config file
        selected_pois = get_active_poi_config()
    
    run_simulation(
        num_residents=args.residents, 
        steps=args.steps, 
        selected_pois=selected_pois,
        parishes_path=args.parishes_path,
        parish_demographics_path=args.parish_demographics,
        create_example_demographics=args.create_example_demographics,
        use_dummy_pois=args.use_dummy_pois,
        selected_parishes=args.parishes,
        list_parishes=args.list_parishes,
        random_distribution=args.random_distribution,
        needs_selection=args.needs_selection,
        movement_behavior=args.movement_behavior
    )