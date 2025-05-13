from environment.city_network import load_city_network
from environment.pois import fetch_pois
from simulation.model import FifteenMinuteCity
from visualization import SimulationAnimator
import matplotlib.pyplot as plt

def run_simulation(num_residents, num_organizations, steps):

    print("Loading Macau's street network...")
    graph = load_city_network("Macau, China")
    pois = fetch_pois(graph)
    

    print(f"Spawning {num_residents} residents...")
    print(f"Spawning {num_organizations} organizations...")
    model = FifteenMinuteCity(graph, pois, num_residents=num_residents, num_organizations=num_organizations)
    

    print("Starting simulation...")
        # Set up interactive mode
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 10))
        # Initialize animator
    animator = SimulationAnimator(model, graph, ax=ax)
    animator.initialize()  # Draw initial state
    for step in range(steps):
        model.step()
        animator.update(step)

        plt.pause(0.1)
        print(f"Step {step + 1}/{steps}", end="\r")
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep window open at end
      # Keep window open

if __name__ == "__main__":
    run_simulation(num_residents=10, num_organizations=3, steps=50)