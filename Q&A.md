# Q&A for the 15-Minute City Agent-Based Model

This document addresses anticipated questions regarding the Agent-Based Model (ABM), its technical implementation, and its potential as a tool for urban planning and policy-making, with a focus on alignment with the UN's Sustainable Development Goals (SDGs).

---

### **Q1: What about the data (geographical and demographical)? How do you ensure it can be trusted, and is it easily doable to extend it to other cities than Macau?**

**A:** This is a foundational question for any model striving for real-world relevance. Our approach to data is two-pronged, balancing accessibility with authority, and is designed for global scalability.

*   **Geographical Data (The Urban Fabric):**
    *   **Source:** We use [OpenStreetMap (OSM)](https://www.openstreetmap.org/) via the `OSMnx` library. OSM is the "Wikipedia of maps," a global, community-driven database of roads, buildings, and Points of Interest (POIs).
    *   **Trust & Verification:** The primary strength of OSM is its global coverage and detail. However, its accuracy can vary by region, as it relies on volunteer contributions. For policy-making applications, we recommend a hybrid approach: using OSM as a baseline and augmenting or cross-validating it with official municipal GIS data (e.g., from a city's planning department) where available. This ensures we have a robust, verified representation of the urban environment.
    *   **Scalability (SDG 11 & 17):** This is where OSM excels. The model can be extended to virtually any city in the world simply by changing the city name in the input arguments (e.g., `--city "Lisbon, Portugal"`). The process is automated. This scalability is key to supporting **SDG 11 (Sustainable Cities and Communities)** globally and fosters **SDG 17 (Partnerships for the Goals)** by leveraging open, collaborative data platforms. The `README.md` provides a detailed guide on this process.

*   **Demographical Data (The Human Fabric):**
    *   **Source:** For Macau, we use a detailed `parish_demographic.json` file derived from official census data. This includes hierarchical distributions of age, gender, and education.
    *   **Trust & Verification:** Demographic data should always come from the most authoritative source possible, typically a national or municipal statistics office. For any new city, the gold standard is to acquire the latest census data at the finest-grained geographical level available (e.g., district, census tract).
    *   **Scalability:** When extending to a new city, the model can run without demographic data, but agents will have default, non-representative characteristics. To achieve a high-fidelity simulation, a city-specific demographic file must be created. This is the most significant data-sourcing task when onboarding a new city, but it is essential for generating meaningful, policy-relevant insights.

---

### **Q2: What is the best number of agents to choose? Is it possible to initialize them more precisely than on nodes of the graph? What kind of data would be needed for that?**

**A:** The number of agents represents a trade-off between computational performance and statistical validity. More precise initialization is certainly possible and represents a key avenue for improving model realism.

*   **Number of Agents:**
    *   There is no single "best" number. The choice depends on the research question and available computing power.
    *   **For Exploratory Analysis:** A few hundred to a few thousand agents can reveal emergent patterns and test model logic quickly.
    - **For Statistical Significance:** To generate results that are representative of a real city, a larger sample is needed. A common practice is to create a 1% or 10% sample of the actual population, ensuring the demographic distribution of the agents matches the census data. The current model's proportional generation (`calculate_proportional_distribution` in `main.py`) is the first step in achieving this.

*   **Precision of Initialization:**
    *   **Current Method:** Agents are initialized at the nearest network node to a randomly selected point within their assigned parish. This is a good approximation but doesn't capture the true distribution of housing.
    *   **Improving Precision:** To initialize agents at their actual homes, we would need **building footprint data** with residential tags. This data specifies the location and geometry of residential buildings.
    *   **Required Data:** The ideal dataset would be a GIS shapefile (or a similar format from OSM) containing polygons for all buildings, with an attribute identifying each building's use (e.g., `residential`, `commercial`, `mixed-use`). With this data, we could:
        1.  Identify all residential buildings within a given parish.
        2.  Distribute agents among these buildings, potentially weighted by the building's size or number of units (if available).
        3.  Place each agent at the actual location of their assigned building.
    *   This would be a significant step up in spatial accuracy, directly impacting the realism of accessibility and travel time calculations, which is critical for **SDG 11**.

---

### **Q3: What is the difference, from a philosophical point of view, between a needs update and a movement decision? How can it be refined using LLMs?**

**A:** This question touches the core of agent-based modeling philosophy: capturing the essence of human decision-making.

*   **Philosophical Difference:**
    *   **Needs Update (Internal State):** This represents an agent's *endogenous*, internal state of being. It's the accumulation of physiological or psychological drivers over time—getting hungrier, feeling a desire for social interaction. It is passive and reflects the agent's current condition. In our model, this is handled by the `increase_needs_over_time` method.
    *   **Movement Decision (External Action):** This is the *exogenous*, deliberate action an agent takes to address its internal state. It's the cognitive process of evaluating needs, perceiving the environment, and choosing a course of action to achieve a goal (e.g., "I am hungry, so I will go to a restaurant"). This is the agent's active response to its state and environment. This is handled by `choose_movement_target`.

    The core difference is between **being (a state)** and **doing (an action)**. Traditional ABMs model the link between them with simple rules (e.g., `if hunger > 50, find food`).

*   **Refinement with Large Language Models (LLMs):**
    *   LLMs offer a paradigm shift from rule-based logic to *generative, goal-oriented reasoning*. Instead of a simple `if-then` rule, an LLM can synthesize a much richer context.
    *   **Implementation Concept:** The `choose_movement_target` method could be transformed into a prompt-based query to an LLM. At each decision point, the agent would "think" by sending a prompt like this:
        > "I am a 35-year-old resident of the Taipa parish. My current needs are: hunger: 75, recreation: 40, social: 60. It is 12:30 PM on a Saturday. The weather is sunny. My available budget is moderate. My personality is introverted. My previous action was working from home. Based on this, what is my most likely next action and destination?"
    *   The LLM's response would be a more nuanced, human-like plan ("You'd probably grab a quiet lunch at a nearby cafe") rather than a mechanistic choice. This allows for more complex, emergent behaviors that are hard to program with explicit rules, advancing our ability to model human societies and innovate towards **SDG 9 (Industry, Innovation, and Infrastructure)**.

---

### **Q4: What outputs can this tool provide? Would it be possible to create heatmaps or calculate indices? How can this tool help policy-making? Can we simulate heatwaves or typhoons?**

**A:** The tool is designed for flexible outputs to directly support evidence-based policy. It can absolutely be extended for advanced analytics and crisis simulation.

*   **Current Outputs:**
    1.  **Live Visualization:** Real-time animation of agent movement.
    2.  **Console Summary:** End-of-run statistics on travel (total trips, average distance/time).
    3.  **Detailed JSON Report:** A rich file (`simulation_report.json`) containing the final state and location history of every agent, which is the raw data for all further analysis.

*   **Advanced Outputs (Yes, all are possible):**
    *   **Heatmaps:** By aggregating the location data from the JSON report over time, we can easily generate heatmaps showing which areas or POIs are most frequented, identifying potential overcrowding or areas of high activity.
    *   **Indices:** We can compute critical policy-relevant indices. A prime example would be a **"15-Minute City Score"** for each parish, calculated by measuring the average travel time from every home to the nearest essential POI (e.g., grocery, clinic, school, park). This score would provide a quantifiable metric of service accessibility.

*   **Helping Policy-Making (SDG 11 & 13):**
    This tool acts as a **"policy sandbox"** or **"digital twin,"** allowing policymakers to test interventions *in silico* before implementing them in the real world. They can ask:
    *   *"What if we build a new clinic in this neighborhood? How would that improve the parish's Healthcare Accessibility Score?"*
    *   *"If we pedestrianize this street, how does it affect travel times and POI visits in the area?"*
    *   *"What is the impact of a new mixed-use housing development on the local POI load?"*

*   **Simulating Crises (Heatwaves/Typhoons):**
    Yes. This is an advanced but critical use case, directly relevant to **SDG 13 (Climate Action)**.
    *   **Heatwave Simulation:** We could add logic where, if the model's "weather" state is a heatwave:
        -   Elderly agents have a higher probability of staying home.
        -   All agents' need for "recreation" shifts towards indoor venues or shaded parks/waterfronts.
        -   Travel behavior changes (e.g., shorter, less frequent trips).
    *   **Typhoon Simulation:** We could temporarily remove nodes or edges from the street network to simulate flooded or blocked roads and observe how residents adapt their travel routes and which areas become isolated.

---

### **Q5: How can the interactions between the agents be made more realistic? What kind of behavior could we assess?**

**A:** Enhancing agent-to-agent interaction is the next frontier for this model, moving from a collection of individuals to a simulated society.

*   **Making Interactions More Realistic:**
    The current model includes a `social_network` attribute, which is the technical foundation for deeper interactions. We can build on this in several ways:
    1.  **Social Influence:** Agents could "recommend" POIs they enjoyed to others in their social network, influencing their friends' choices.
    2.  **Coordinated Activities:** Agents could decide to meet up. For example, two friends could decide to visit a park *together*, requiring them to solve a coordination problem (choosing a time and place).
    3.  **Information/Rumor Spreading:** We could model how information (e.g., "the supermarket on X street is having a sale," or misinformation) spreads through the social network, affecting collective behavior.
    4.  **Economic Interactions:** Introduce direct agent-to-agent transactions or have agents be employees at POIs owned by other agents.

*   **Behaviors to Assess:**
    With more realistic interactions, we can study powerful emergent phenomena:
    *   **Formation of Social Hubs:** Do certain parks, cafes, or community centers naturally emerge as popular meeting spots for social groups?
    *   **Community Resilience:** In a crisis simulation (like a typhoon), how do social networks affect a community's ability to cope? Do agents with stronger social ties fare better?
    *   **Equity and Social Segregation:** Do social networks tend to form within the same parish or demographic group (e.g., income, education)? Does this lead to social segregation in the use of urban space? This is a critical question for **SDG 10 (Reduced Inequalities)**.

---

### **Q6: What parameters would be interesting to play with for policymakers?**

**A:** The model is a sandbox, and policymakers can "play with" various parameters that correspond to real-world policy levers. Here are the most impactful ones:

1.  **POI Distribution (Urban Planning & Zoning):**
    *   **Parameter:** The list of POIs and their locations.
    *   **Policy Question:** *What is the impact of opening a new school in Parish A? Or closing a hospital in Parish B?* By adding or removing POIs, they can directly measure the effect on accessibility and agent travel patterns.

2.  **Infrastructure Changes (Transportation Planning):**
    *   **Parameter:** The `graph` object representing the street network.
    *   **Policy Question:** *If we add a new pedestrian bridge here or make a street one-way, how does that change neighborhood connectivity?* Modifying the network graph allows for testing infrastructural changes.

3.  **Demographic Scenarios (Long-Term Strategic Planning):**
    *   **Parameter:** The input demographic data.
    *   **Policy Question:** *Our city's population is projected to age significantly over the next 20 years. How will this impact the demand for healthcare services and accessible public spaces?* Policymakers can run simulations with future demographic scenarios to anticipate needs.

4.  **POI Operating Parameters (Economic & Regulatory Policy):**
    *   **Parameter:** `open_hours` or `capacity` on POI agents.
    *   **Policy Question:** *What is the effect of extending library hours or providing incentives for grocery stores to increase their capacity in underserved areas?*

5.  **Crisis Scenarios (Emergency Management):**
    *   **Parameter:** Environmental flags (e.g., `is_heatwave=True`) or temporary graph modifications (e.g., flooded roads).
    *   **Policy Question:** *Which communities are most vulnerable during a flood? Where should we place emergency cooling centers during a heatwave?*

By manipulating these parameters, the model transforms from an observational tool into an **interactive decision-support system**, empowering policymakers to build more sustainable, equitable, and resilient cities, in direct alignment with **SDG 11**. 