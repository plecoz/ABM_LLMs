from environment.city_network import load_city_network
from environment.pois import fetch_pois, filter_pois
from simulation.model import FifteenMinuteCity
from visualization import SimulationAnimator
import matplotlib.pyplot as plt
import sys
import os
import geopandas as gpd
import json

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
DEFAULT_PARISHES_PATH = "C:/Users/pierr/OneDrive/Documents/Stage Macau/shapefiles/macau_shapefiles/macau_districts.gpkg"

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

def run_simulation(num_residents, num_organizations, steps, selected_pois=None, parishes_path=None, parish_demographics_path=None, create_example_demographics=False):
    """
    Run the 15-minute city simulation.
    
    Args:
        num_residents: Number of resident agents to create
        num_organizations: Number of organization agents to create
        steps: Number of simulation steps to run
        selected_pois: List of POI types to include (e.g., ['bank', 'police', 'school', 'hospital', 'fire_station'])
                      If None, all POIs will be included
        parishes_path: Path to the shapefile with Macau parishes/districts
        parish_demographics_path: Path to JSON file with parish-specific demographics
        create_example_demographics: Whether to create example demographics based on parishes
    """
    print("Loading Macau's street network...")
    graph = load_city_network("Macau, China")
    
    # Load parishes data
    parishes_gdf = load_parishes(parishes_path)
    
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
    
    # Fetch POIs with selected types
    pois = fetch_pois(graph, selected_pois=selected_pois)
    
    print(f"Spawning {num_residents} residents...")
    print(f"Spawning {num_organizations} organizations...")
    model = FifteenMinuteCity(
        graph, 
        pois, 
        num_residents=num_residents, 
        num_organizations=num_organizations,
        parishes_gdf=parishes_gdf,
        parish_demographics=parish_demographics
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
    
    for step in range(steps):
        model.step()
        animator.update(step)
        
        plt.pause(0.1)
        print(f"Step {step + 1}/{steps}", end="\r")
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep window open at end

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run the 15-minute city simulation with Macau parishes')
    parser.add_argument('--essential-only', action='store_true', help='Only use essential services POIs')
    parser.add_argument('--all-pois', action='store_true', help='Use all available POI types')
    parser.add_argument('--residents', type=int, default=100, help='Number of resident agents')
    parser.add_argument('--organizations', type=int, default=3, help='Number of organization agents')
    parser.add_argument('--steps', type=int, default=50, help='Number of simulation steps')
    parser.add_argument('--parishes-path', type=str, help='Path to parishes/districts shapefile')
    parser.add_argument('--parish-demographics', type=str, help='Path to parish-specific demographics JSON file')
    parser.add_argument('--create-example-demographics', action='store_true', help='Create example parish demographics')
    
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
        num_organizations=args.organizations, 
        steps=args.steps, 
        selected_pois=selected_pois,
        parishes_path=args.parishes_path,
        parish_demographics_path=args.parish_demographics,
        create_example_demographics=args.create_example_demographics
    )