# ABM_LLMs

# 15-Minute City Simulation for Macau

This agent-based model simulates a 15-minute city concept applied to Macau, China. It models residents, points of interest (POIs), and their interactions within the urban environment.

## Features

- **Agent-Based Modeling**: Simulates individual residents and POIs as agents
- **Geographic Data**: Uses real OpenStreetMap data for Macau
- **POI Categories**:
  - Daily Living: Grocery stores, banks, restaurants, barber shops, post offices
  - Healthcare: Hospitals, clinics, pharmacies
  - Education: Kindergartens, primary schools, secondary schools
  - Entertainment: Parks, public squares, libraries, museums, etc.
  - Transportation: Bus stops (displayed as small black diamonds)
- **Waiting Time Simulation**: Models waiting times at specific POI types during peak hours

## Waiting Time Feature

The simulation includes a realistic waiting time model for specific POI types:
- **Applicable POIs**: Grocery stores, banks, restaurants, barber shops, and pharmacies
- **Peak Hours**:
  - Morning: 7-9 AM
  - Lunch: 12-2 PM
  - Evening: 5-7 PM
- **Waiting Time Calculation**: Based on:
  - Number of visitors
  - POI capacity
  - Time of day (peak hours have 50% longer wait times)
  - Service time specific to each POI type

Residents consider waiting times when choosing which POIs to visit, balancing distance, waiting time, and popularity.

## Running the Simulation

```bash
# Basic simulation
python main.py

# With dummy POIs for testing
python main.py --use-dummy-pois

# Hide waiting time indicators
python main.py --no-waiting-times

# Hide daily living POIs
python main.py --hide-daily-living

# Use only essential services
python main.py --essential-only
```

## Command Line Options

- `--residents N`: Set number of resident agents (default: 100)
- `--steps N`: Set number of simulation steps (default: 50)
- `--use-dummy-pois`: Use generated dummy POIs instead of fetching from OpenStreetMap
- `--hide-daily-living`: Hide daily living POIs in visualization
- `--no-waiting-times`: Hide waiting time indicators
- `--essential-only`: Only use essential services POIs
- `--all-pois`: Use all available POI types