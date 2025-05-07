from agents import Resident, POI
from environment import load_city_network, fetch_pois
from simulation import FifteenMinuteCity


graph = load_city_network(place_name="Macau")  # Explicitly set Macau
pois = fetch_pois(graph, place_name="Macau")

# Run simulation
model = FifteenMinuteCity(graph, pois)
for _ in range(10):  # Simulate 10 steps
    model.step()