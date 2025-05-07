from agents import Resident, POI
from environment import load_city_network, fetch_pois
from simulation import FifteenMinuteCity


# Initialize city and POIs
graph = load_city_network(place_name="Paris, France")
pois = fetch_pois(graph)

# Run simulation
model = FifteenMinuteCity(graph, pois)
for _ in range(10):  # Simulate 10 steps
    model.step()