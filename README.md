# 15-Minute City Agent-Based Model with LLM Integration

This agent-based model (ABM) simulates urban life within the "15-minute city" concept, where residents can meet their daily needs within a short travel distance from home. The simulation is built using the Mesa framework and leverages real-world geographic data from OpenStreetMap. It features **LLM-powered agent decision-making** using GPT models through the Concordia framework for realistic, explainable behavior.

## Core Features

- **LLM-Powered Agents**: Resident agents use Large Language Models (GPT-4.1-mini) for intelligent decision-making about where to go and what to do
- **Simple, Explainable Personas**: Agents have clear demographic profiles (age, household type, economic status) that influence their behavior transparently
- **Realistic Environment**: Builds the simulation environment from real-world city street networks and Points of Interest (POIs) via OSMnx
- **Dynamic Path Selection**: LLM agents can choose between multiple routes based on their personality and preferences
- **Parish/District-Based Simulation**: Can run simulations for an entire city or focus on specific, user-selected parishes (Macau) or districts
- **Advanced Visualization**:
    - Live animation of agent movements on the city map
    - Color-coded POI categories and parish boundaries
    - Dynamically adjusting scale bar and a north arrow for geographic context
- **Data Output**: Generates detailed JSON reports of simulation runs and agent travel metrics
- **Extensible & Modular**: Designed with a clear project structure to easily add new agent behaviors, cities, or data sources

---

## Project Structure

```
.
├── agents/                    # Agent class definitions
│   ├── fifteenminutescity/
│   │   ├── resident.py        # Main Resident agent with LLM integration
│   │   ├── poi.py             # Point of Interest (POI) agent
│   │   └── persona_memory_modules.py # Simple persona system
├── brains/
│   └── concordia_brain.py     # LLM brain wrapper for Concordia integration
├── config/
│   └── llm_config.py          # LLM configuration (model, API keys, etc.)
├── data/                      # Shapefiles and demographic data
│   ├── macau_shapefiles/      # Geographic data for Macau
│   └── demographics_macau/    # Demographic distributions for agent generation
├── environment/               # Scripts for building the simulation world
│   ├── fifteenminutescity/
│   │   ├── city_network.py    # Street network handling
│   │   └── pois.py            # POI fetching and filtering
├── simulation/                # Core simulation logic
│   └── fifteenminutescity/
│       └── fifteenminutescity_model.py # Main Mesa model with persona assignment
├── visualization/             # Visualization and animation code
│   └── animator.py            # Matplotlib animation manager
├── main.py                    # Main entry point
└── README.md                  # This file
```

---

## LLM Integration Setup

### 1. Configure Your LLM Connection

Edit `config/llm_config.py`:

```python
# Model configuration
PROVIDER = "custom_openai"  # For custom endpoints
MODEL_NAME = "gpt-4o-mini"  # Or "gpt-4.1-mini", "gpt-3.5-turbo"
API_KEY = "your-api-key-here"
BASE_URL = "https://your-api-endpoint.com/v1"  # Optional: custom endpoint
```

### 2. Set Environment Variable (Recommended)

    ```bash
export OPENROUTER_API_KEY="your-api-key-here"
    ```

### 3. Install Dependencies

    ```bash
pip install mesa geopandas osmnx matplotlib numpy pandas openai
    ```

---

## How to Run the Simulation

### Basic Example with LLM Agents

```bash
python main.py --residents 10 --steps 100 --movement-behavior llms
```

This runs a simulation with 10 LLM-powered residents for 100 steps (minutes).

### Command-Line Arguments

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `--residents` | Number of resident agents | 100 | `--residents 50` |
| `--steps` | Number of simulation steps (minutes) | 100 | `--steps 200` |
| `--movement-behavior` | Agent movement behavior | 'need-based' | `--movement-behavior llms` |
| `--needs-selection` | How agents generate needs | 'random' | `--needs-selection llms` |
| `--city` | City to simulate | 'Macau, China' | `--city "Barcelona, Spain"` |
| `--parishes` | Specific parishes/districts | (all) | `--parishes "Taipa" "Coloane"` |
| `--save-json-report` | Save detailed JSON report | (not set) | `--save-json-report` |

### Movement Behavior Options

- **`need-based`**: Traditional rule-based agent behavior
- **`llms`**: LLM-powered decision making for both target selection and path choice
- **`random`**: Random movement for testing

### LLM-Powered Simulation Example

```bash
python main.py --residents 20 --steps 150 --movement-behavior llms --parishes "Taipa" --save-json-report
```

This creates 20 LLM agents in Taipa parish, runs for 150 minutes, and saves a detailed report.

---

## Agent Persona System

### Simple, Explainable Demographics

Each agent has a clear, interpretable persona based on three key characteristics:

```python
# Example personas sent to LLM
"Resident 72y: Age: 72, Household: elderly, Income: middle, Location: Taipa"
"Resident 35y: Age: 35, Household: family, Income: high, Location: São Lourenço"  
"Resident 22y: Age: 22, Household: single, Income: low, Location: Coloane"
```

### Persona Categories

- **Household Type**: `elderly`, `family`, `single`
- **Economic Status**: `low`, `middle`, `high`
- **Age**: Actual age number
- **Location**: Parish/district name

### How Personas Influence Behavior

The LLM interprets these demographics naturally:

- **Elderly agents** → Higher healthcare needs, prefer familiar routes
- **Family agents** → More shopping trips, efficiency-focused decisions  
- **Young singles** → More social/recreation activities, willing to try new routes
- **Low income** → More cost-conscious, fewer recreational trips
- **High income** → More varied activities, less concerned about distance

---

## Key Implementation Details

### LLM Decision Making

When `--movement-behavior llms` is used:

1. **Target Selection**: LLM chooses where to go based on agent needs and persona
2. **Path Selection**: If multiple routes exist, LLM chooses based on preferences
3. **JSON Responses**: Agents return structured decisions like `{"action": "move", "target_poi_id": 147}`

### Realistic Agent Placement

- **Building-Based Homes**: Agents start in real residential buildings
- **Access Time**: Walking time from building to street network is calculated
- **Network-Based Movement**: All travel uses real street networks

### Simulation Time Steps

Each step = 1 minute of simulation time:
1. **Needs Update**: Agent needs increase naturally
2. **LLM Decision**: If idle, agent asks LLM what to do next
3. **Movement**: Agent travels toward chosen destination
4. **Activity**: Agent performs activities at POIs

---

## Explainable AI Features

### Transparent Decision Factors

- **Clear Demographics**: Easy to understand what influences each agent
- **Simple Categories**: No hidden complexity or black-box personality generation
- **Direct Causality**: Demographics → Needs → LLM Decision → Behavior

### Debug and Analysis

- **Decision Logging**: See exactly what the LLM decided and why
- **Path Selection Analysis**: Track when agents choose non-optimal routes
- **Persona Tracing**: Follow how demographics influence specific decisions

### Research-Friendly

- **Reproducible**: Same demographics produce consistent behavior patterns
- **Interpretable**: Can explain any agent's behavior in simple terms
- **Modifiable**: Easy to test how changing demographics affects outcomes

---

## Performance and Rate Limiting

### API Usage

- **Initialization**: 1 API call per agent (sets persona)
- **Runtime**: 1 API call per decision (when agent chooses new destination)
- **Path Selection**: Additional calls when multiple routes available

### Optimization Tips

1. **Use GPT-4o-mini**: Faster and higher rate limits than GPT-4
2. **Fewer Agents**: Start with 10-20 agents to avoid rate limits
3. **Shorter Simulations**: 100-200 steps for initial testing
4. **Monitor Logs**: Watch for 429 (rate limit) errors

---

## Supported Cities

The simulation works with any city via OpenStreetMap, with enhanced support for:

1. **Macau, China** (Default)
   - Detailed parish boundaries and demographic data
   - All parishes: Santo Antonio, São Lázaro, São Lourenço, Sé, Nossa Senhora de Fátima, Taipa, Coloane

2. **Barcelona, Spain**
   - District boundaries available
   - Use `--list-parishes` to see districts

3. **Hong Kong, China**
   - Administrative district support
   - Use `--list-parishes` for available areas

---

## Adding New Cities

```bash
# Basic simulation for any city
python main.py --city "Tokyo, Japan" --movement-behavior llms

# With caching for repeated runs
python main.py --city "Tokyo, Japan" --save-network data/tokyo_network.pkl --save-pois data/tokyo_pois.pkl

# Subsequent runs (much faster)
python main.py --city "Tokyo, Japan" --load-network data/tokyo_network.pkl --load-pois data/tokyo_pois.pkl --movement-behavior llms
```

---

## Output and Analysis

### Live Visualization
- Real-time agent movement on city map
- Color-coded POIs and parish boundaries
- Agent trails and current locations

### Console Output
- Agent decision summaries
- Path selection statistics  
- API usage and performance metrics

### JSON Reports
    ```bash
--save-json-report
```
Creates detailed reports with:
- Complete agent state histories
- Decision logs and reasoning
- Path selection analysis
- Performance metrics

---

## Future Enhancements

### Short Term
- **Multi-modal transport**: LLM agents choosing between walking/transit
- **Social interactions**: Agents influencing each other's decisions
- **Dynamic POI capacity**: Agents avoiding crowded locations

### Long Term  
- **Economic modeling**: Budget constraints affecting LLM decisions
- **Learning agents**: Agents that remember and adapt behavior
- **Multi-agent coordination**: Group decision making

---

## Contributing

This simulation is designed for research into explainable AI in urban modeling. Contributions welcome for:

- New LLM providers and models
- Additional demographic factors
- Enhanced decision logging and analysis
- Performance optimizations

---

## Research Applications

- **Urban Planning**: Test how different demographics respond to city changes
- **Policy Analysis**: Model impact of new services or infrastructure  
- **AI Explainability**: Study how simple demographics drive complex behavior
- **Transportation**: Analyze route choice and travel patterns
- **Social Simulation**: Understand community dynamics and interactions
