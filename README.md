# ABM_LLMs

# 15-Minute City Simulation for Macau & Hong Kong

This agent-based model simulates a 15-minute city concept applied to Macau and Hong Kong. It models residents, points of interest (POIs), and their interactions within the urban environment.

## Features

- **Agent-Based Modeling**: Simulates individual residents and POIs as agents
- **Geographic Data**: Uses real OpenStreetMap data for Macau and Hong Kong
- **POI Categories**:
  - Daily Living: Grocery stores, banks, restaurants, barber shops, post offices
  - Healthcare: Hospitals, clinics, pharmacies
  - Education: Kindergartens, primary schools, secondary schools
  - Entertainment: Parks, public squares, libraries, museums, etc.
  - Transportation: Bus stops (displayed as small black diamonds)
- **Waiting Time Simulation**: Models waiting times at specific POI types during peak hours
- **Flexible City Support**: Run the simulation in either Macau or Hong Kong with the `--city` argument
- **Demographic Realism**: Macau supports parish-based demographic distributions; Hong Kong uses random distribution
- **Parish/District Filtering**: Filter simulation to specific parishes (Macau) or districts (Hong Kong)
- **Smooth Animation**: Residents move smoothly along the street network between POIs

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

## Agent Types and Functionalities

### Resident Agents
- **Representation**: Each resident is an individual agent with demographic and behavioral properties.
- **Key Properties**:
  - `age`, `gender`, `income`, `education`, `employment_status`, `household_type`
  - `home_node`: The node in the street network where the agent lives
  - `current_node`: The agent's current location
  - `energy`: Depletes with movement, recharges at home
  - `mobility_mode`: Currently set to "walk" (speed varies by age)
  - `parish`: The parish/district the agent belongs to
  - `accessible_nodes`: Nodes within 1km of home (for activity selection)
  - `social_network`: List of other resident IDs (social ties)
  - `activity_preferences`: Dict of activity weights (can be customized)
- **Behaviors**:
  - **Movement**: Chooses POIs to visit based on preferences, distance, and waiting time
  - **Travel**: Moves along the shortest path to POIs, with travel time based on speed and distance
  - **Energy Management**: Loses energy when away from home, recharges at home
  - **Social Interaction**: Forms social networks and can communicate with nearby agents
  - **Needs**: Tracks dynamic needs (hunger, social, recreation, shopping, healthcare, education)
  - **Demographics**: Properties sampled from global or parish-specific distributions

### POI Agents
- **Representation**: Each POI (shop, hospital, school, etc.) is a static agent at a network node
- **Key Properties**:
  - `poi_type`: Type of POI (e.g., 'supermarket', 'hospital')
  - `category`: Category (daily_living, healthcare, education, entertainment, transportation)
  - `capacity`: Max simultaneous visitors
  - `open_hours`: Opening hours (default 8am-8pm)
  - `visitors`: Set of current resident IDs at the POI
  - `waiting_time`: Calculated based on visitors, capacity, and peak hours
- **Behaviors**:
  - **Waiting Time Calculation**: For certain POI types, waiting time increases during peak hours and with more visitors
  - **Visitor Tracking**: Updates the set of current visitors each step
  - **Open/Closed Logic**: Only available during open hours

## Model Steps: Simulation Loop

Each simulation step (typically 15 minutes or 1 hour, depending on configuration) proceeds as follows:

1. **Time Update**: Model advances the hour of day and day of week
2. **Agent Activation**: All agents (residents and POIs) are activated in random order
3. **Resident Step**:
   - If traveling, decrement travel time; if arrived, update location and POI visitor list
   - If not traveling, select a POI to visit based on preferences and needs
   - Calculate travel time and start moving if a new destination is chosen
   - Update energy (deplete if away from home, recharge if at home)
   - If energy is depleted, return home
   - Interact with nearby agents (social network formation, communication)
4. **POI Step**:
   - Update current visitors
   - Recalculate waiting time if applicable
5. **Data Collection**: Model records agent states and statistics for analysis
6. **Visualization**: The current state is rendered, showing agent positions, POIs, and waiting times

## City Selection: Macau vs Hong Kong

- **Macau** (default):
  - Supports parish-based filtering and proportional demographic distribution
  - Can use real parish demographic data (JSON)
  - Command: `python main.py --city macau`
- **Hong Kong**:
  - Uses district boundaries for visualization
  - Residents are distributed randomly (no parish/district demographic data yet)
  - Command: `python main.py --city hongkong`

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