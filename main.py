from environment.city_network import load_city_network
from environment.pois import fetch_pois
from simulation.model import FifteenMinuteCity
from visualization import SimulationAnimator

def run_simulation(num_agents=10, steps=50):
    # 1. Load Macau's infrastructure
    print("⏳ Loading Macau's street network...")
    graph = load_city_network("Macau, China")
    pois = fetch_pois(graph)
    
    # 2. Initialize model with 10 agents
    print(f"🚶 Spawning {num_agents} residents...")
    model = FifteenMinuteCity(graph, pois, num_agents=num_agents)
    
    # 3. Run with visualization
    print("🌆 Starting simulation...")
    animator = SimulationAnimator(model, graph)
    for step in range(steps):
        model.step()
        animator.update(step)
        print(f"⏱️ Step {step + 1}/{steps}", end="\r")
    
    plt.show()  # Keep window open

if __name__ == "__main__":
    run_simulation(num_agents=10, steps=50)