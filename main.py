from environment.city_network import load_city_network
from environment.pois import fetch_pois, filter_pois
from simulation.model import FifteenMinuteCity
from visualization import SimulationAnimator
import matplotlib.pyplot as plt
import sys
import os

# Import POI configuration
try:
    from config.poi_config import get_active_poi_config
except ImportError:
    # Create config directory if it doesn't exist
    if not os.path.exists('config'):
        os.makedirs('config')
    print("POI configuration not found. Using default configuration (all POIs).")
    get_active_poi_config = lambda: None

def run_simulation(num_residents, num_organizations, steps, selected_pois=None):
    """
    Run the 15-minute city simulation.
    
    Args:
        num_residents: Number of resident agents to create
        num_organizations: Number of organization agents to create
        steps: Number of simulation steps to run
        selected_pois: List of POI types to include (e.g., ['bank', 'police', 'school', 'hospital', 'fire_station'])
                      If None, all POIs will be included
    """
    print("Loading Macau's street network...")
    graph = load_city_network("Macau, China")
    
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
    model = FifteenMinuteCity(graph, pois, num_residents=num_residents, num_organizations=num_organizations)
    

    print("Starting simulation...")
    # Set up interactive mode
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Initialize animator
    animator = SimulationAnimator(model, graph, ax=ax)
    
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
    # Get POI configuration from command line if provided
    if len(sys.argv) > 1 and sys.argv[1] == "--essential-only":
        from config.poi_config import ESSENTIAL_SERVICES_ONLY
        selected_pois = ESSENTIAL_SERVICES_ONLY
    elif len(sys.argv) > 1 and sys.argv[1] == "--all-pois":
        selected_pois = None
    else:
        # Use configuration from config file
        selected_pois = get_active_poi_config()
    
    run_simulation(num_residents=100, num_organizations=3, steps=50, selected_pois=selected_pois)