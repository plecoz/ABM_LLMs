# 15-Minute City Agent-Based Model

This agent-based model (ABM) simulates urban life within the "15-minute city" concept, where residents can meet their daily needs within a short travel distance from home. The simulation is built using the Mesa framework and leverages real-world geographic data from OpenStreetMap. It is highly configurable and can be adapted to simulate different cities, starting with Macau and Hong Kong.

## Core Features

- **Dynamic Agent Behavior**: Simulates resident agents with unique demographic profiles, needs, and daily behaviors.
- **Realistic Environment**: Builds the simulation environment from real-world city street networks and Points of Interest (POIs) via OSMnx.
- **Hierarchical Demographics (Macau)**: Generates agents based on a detailed, hierarchical demographic system: `Parish -> Age Class -> Gender -> Education`.
- **Employment & Commuting**: Models employment status based on education level (for Macau) and simulates commuting behavior.
- **Parish/District-Based Simulation**: Can run simulations for an entire city or focus on specific, user-selected parishes (Macau) or districts.
- **Configurable Scenarios**: Supports different agent need generation models (`random`, `maslow`, etc.) and movement behaviors.
- **Advanced Visualization**:
    - Live animation of agent movements on the city map.
    - Color-coded POI categories and parish boundaries.
    - Dynamically adjusting scale bar and a north arrow for geographic context.
- **Data Output**: Generates a detailed JSON report of the simulation run and a summary of agent travel metrics.
- **Extensible & Modular**: Designed with a clear project structure to easily add new agent behaviors, cities, or data sources.

---

## Project Structure

```
.
├── agents/               # Agent class definitions
│   ├── resident.py       # Defines the Resident agent's logic and properties
│   └── poi.py            # Defines the Point of Interest (POI) agent
├── data/                 # Shapefiles and demographic data
│   ├── macau_shapefiles/ # Geographic data for Macau
│   └── parish_demographic.json # Detailed demographic data for Macau parishes
├── environment/          # Scripts for building the simulation world
│   ├── city_network.py   # Handles fetching and loading the street network
│   └── pois.py           # Handles fetching and filtering POIs
├── simulation/           # Core simulation logic
│   └── model.py          # The main Mesa model orchestrating the simulation
├── visualization/        # Visualization and animation code
│   └── animator.py       # Manages the Matplotlib animation
├── main.py               # Main entry point to run the simulation
└── README.md             # This file
```

---

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install mesa geopandas osmnx matplotlib numpy pandas
    ```

---

## How to Run the Simulation

The simulation is run from the command line via `main.py`.

### Basic Example

This will run a default simulation for **Macau** with 100 residents for 100 steps.

```bash
python main.py --residents 100 --steps 100
```

### Command-Line Arguments

You can customize the simulation with the following arguments:

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `--residents` | Number of resident agents. | 100 | `--residents 500` |
| `--steps` | Number of simulation steps to run. | 100 | `--steps 200` |
| `--city` | Name of the city to simulate. Currently supports parishes data for Macau, Barcelona, and Hong Kong. | 'Macau, China' | `--city "Barcelona, Spain"` |
| `--parishes` | A list of specific parishes to run the simulation in (for Macau). | (all) | `--parishes "Taipa" "Coloane"` |
| `--demographics` | Path to the parish demographics JSON file. | `data/parish_demographic.json` | `--demographics path/to/my_data.json` |
| `--save-json-report` | Save a detailed JSON report of the simulation. | (not set) | `--save-json-report` |
| `--random-distribution` | Distribute residents randomly across all parishes, ignoring proportions. | (not set) | `--random-distribution` |
| `--use-dummy-pois` | Use a small set of dummy POIs for faster testing. | (not set) | `--use-dummy-pois` |
| `--list-parishes` | List all available parishes for the selected city and exit. | (not set) | `--list-parishes` |
| `--save-network` | Save the downloaded city network to a file. | (not set) | `--save-network path/to/network.pkl` |
| `--load-network` | Load a pre-saved city network from a file. | (not set) | `--load-network path/to/network.pkl` |
| `--save-pois` | Save the downloaded POIs to a file. | (not set) | `--save-pois path/to/pois.pkl` |
| `--load-pois` | Load pre-saved POIs from a file. | (not set) | `--load-pois path/to/pois.pkl` |

### Running a Focused Simulation

To run a simulation for only the "Taipa" and "Coloane" parishes in Macau with 500 residents and save a report:

```bash
python main.py --residents 500 --steps 150 --parishes "Taipa" "Coloane" --save-json-report
```

To run a simulation for Barcelona's Ciutat Vella district:

```bash
python main.py --city "Barcelona, Spain" --parishes "Ciutat Vella" --residents 500 --steps 150
```

### Supported Cities with Parish/District Data

The simulation currently includes district/parish data for the following cities:

1. **Macau, China** (Default)
   - Includes all parishes: Santo Antonio, So Lzaro, So Loureno, S, Nossa Senhora de Ftima, Taipa, Coloane
   - Includes detailed demographic data and proportional distribution

2. **Barcelona, Spain**
   - Includes all districts with their administrative boundaries
   - Use `--list-parishes` to see available districts

3. **Hong Kong, China**
   - Includes administrative district boundaries
   - Use `--list-parishes` to see available districts

For other cities, the simulation will run without parish/district visualization, but all other features (POIs, resident movement, etc.) will work normally.

---

## Key Concepts & Implementation Details

### Resident Agent

The `Resident` agent is the core of the simulation. Each resident has a rich set of attributes that guide their behavior:

-   **Demographics**: `age`, `age_class`, `gender`, `education`, `income`, `parish`. These are generated from the hierarchical demographic data.
-   **Employment**: For Macau simulations, `employment_status` is probabilistically determined based on the agent's education level. This feature is disabled for other cities.
-   **Needs**: A simple needs system (`hunger`, `social`, `recreation`, etc.) drives the agent's decisions to visit POIs.
-   **Movement**: Agents travel along the real street network. Travel time is calculated in 1-minute steps (assuming an 80-meter walking distance per minute).

### Point of Interest (POI) Agent

POIs are static agents representing real-world locations.

-   **Categorization**: POIs are fetched from OpenStreetMap and categorized into:
    -   `daily_living` (e.g., supermarkets, banks)
    -   `healthcare` (e.g., hospitals, pharmacies)
    -   `education` (e.g., schools, universities)
    -   `entertainment` (e.g., parks, museums)
    -   `transportation` (e.g., bus stops)
-   **Interaction**: Residents visit POIs to satisfy their needs.

### The Simulation Loop: What Happens in a Step?

Each step of the simulation represents **one minute** of time. When the model advances one step, the following events occur in sequence:

1.  **Time Advancement**: The model's internal clock is updated (e.g., from 8:00 AM to 8:01 AM).
2.  **Agent Activation (Random Order)**: The scheduler activates each agent (`Resident` and `POI`) one by one in a shuffled, random order to prevent artifacts from fixed activation patterns.
3.  **Agent `step()` Execution**: Each agent performs its actions for the minute.
    -   **For a `Resident` agent**:
        1.  **Needs Increase**: Basic needs like hunger or recreation slightly increase.
        2.  **Check Activity Status**: The agent checks if it is currently traveling or waiting at a POI.
        3.  **If Traveling**: `travel_time_remaining` is decremented. If it reaches zero, the agent arrives at its destination.
        4.  **If Waiting at a POI**: `waiting_time_remaining` is decremented.
        5.  **If Idle (not traveling or waiting)**: The agent makes a decision.
            - It evaluates its current needs to decide what to do next.
            - It may choose to visit a POI to satisfy a need or decide to return home.
            - Once a destination is chosen, it calculates the path and begins traveling.
    -   **For a `POI` agent**:
        1.  **Update Visitors**: It scans the simulation to see which residents are currently at its location.
        2.  **Update Waiting Time**: It recalculates its internal waiting time based on the number of current visitors and the time of day (peak hours). *Note: This is an internal state; residents do not yet use this information for decision-making.*
4.  **Data Collection**: The `DataCollector` records the state of the model and each agent at the end of the step for later analysis and for the final JSON report.

This cycle repeats for the specified number of steps, creating a dynamic simulation of urban life.

---

## How to Add a New City

The simulation is designed to be city-agnostic. To add and simulate a new city (e.g., "Lisbon, Portugal"), follow these steps:

1.  **Use the `--city` argument**: The most crucial step is to specify the new city in the command line. OSMnx will automatically download the correct street network.
    ```bash
    python main.py --city "Lisbon, Portugal"
    ```

2.  **(Optional) Add a Districts Shapefile**: To visualize the city's districts or parishes and confine the simulation to specific areas, you need a geographic shapefile (e.g., `.gpkg`, `.shp`).
    -   Place the shapefile in the `data/` directory (e.g., `data/lisbon_shapefiles/`).
    -   In `main.py`, update the `DEFAULT_PARISHES_PATH` variable to point to your new file.

3.  **(Optional) Add Demographic Data**: For high-fidelity simulations with realistic agent generation (like in Macau), you can create a demographic JSON file.
    -   Create a file similar to `data/parish_demographic.json` with age, gender, and education distributions for the new city's districts.
    -   Use the `--demographics` argument to point the simulation to your new file.
    -   If no demographic file is provided, residents will be generated with default attributes and distributed randomly.

4.  **(Recommended) Use Caching Arguments**: For new cities, downloading the network and POIs can be slow. Run the simulation once with `--save-network` and `--save-pois`, then use `--load-network` and `--load-pois` for all subsequent runs to speed up initialization significantly.
    ```bash
    # First run (downloads and saves)
    python main.py --city "Lisbon, Portugal" --save-network data/lisbon_network.pkl --save-pois data/lisbon_pois.pkl

    # Subsequent runs (loads from file)
    python main.py --city "Lisbon, Portugal" --load-network data/lisbon_network.pkl --load-pois data/lisbon_pois.pkl
    ```

---

## Unimplemented Features & Future Work

While the simulation is feature-rich, some attributes in the code are placeholders for future development. This means the variables exist on the agents, but the logic to fully use them is not yet implemented. Your contributions are welcome!

-   **POI Capacity and Waiting Times**:
    -   The `poi.py` agent has `capacity` and `current_waiting_time` attributes.
    -   **Current State**: POIs calculate a waiting time based on visitor count, but **residents do not yet consider this information** when deciding which POI to visit. They will travel to a POI regardless of how "full" it is or how long the wait is.
    -   **Future Goal**: Implement logic for residents to query POI waiting times and choose less crowded options.

-   **Agent Energy**:
    -   The `resident.py` agent has commented-out code for an `energy` mechanic.
    -   **Current State**: This feature is **completely inactive**. Residents can travel indefinitely without needing to return home to rest.
    -   **Future Goal**: Activate the energy system to require agents to return home, adding a layer of realism to their daily schedules.

-   **Dynamic Needs & Satisfaction**:
    -   The `dynamic_needs` attribute in `resident.py` is a placeholder.
    -   **Current State**: Needs increase at a constant, linear rate. Visiting a POI satisfies a fixed amount of a need.
    -   **Future Goal**: Develop a more complex system where needs change based on activities, time of day, and social interactions.

-   **Advanced Social Networking**:
    -   Agents have a `social_network` attribute and can identify nearby agents.
    -   **Current State**: The logic for forming and leveraging these social networks for decision-making (e.g., visiting a POI together) is minimal.
    -   **Future Goal**: Build out social behaviors, allowing agents to influence each other's decisions.

-   **Deeper Economic Model**:
    -   Agents have `income` and `employment_status`.
    -   **Current State**: These attributes are purely descriptive and do not affect agent behavior. There is no model for spending money or prices at POIs.
    -   **Future Goal**: Implement an economic layer where agents have budgets and POIs have prices, influencing choices.

-   **Multi-Modal Transport**:
    -   **Current State**: All agents currently walk at a speed determined by their age.
    -   **Future Goal**: Add other transport modes like using public transit (by interacting with `transportation` POIs), or cycling.

---

## Simulation Output

The simulation provides output in three main ways:

1.  **Live Visualization**: A Matplotlib window shows the real-time movement of agents on the city map.
2.  **Console Output**: At the end of the simulation, a summary of agent travel statistics (total trips, average distance, etc.) is printed to the console.
3.  **JSON Report** (optional): If you use the `--save-json-report` flag, a `simulation_report.json` file is created, containing a complete snapshot of every agent's state at the end of the simulation, including their location history.
