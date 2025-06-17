# LLM Integration Guide for ABM Healthcare Policy Simulation

## Overview

This guide explains the integration of Large Language Model (LLM) capabilities into the Agent-Based Model (ABM) for healthcare policy simulation. The integration enables more realistic and sophisticated agent behavior through persona-driven decision making and emotional state modeling.

## Components Integrated

### 1. Persona Memory Module (`agents/persona_memory_modules.py`)

The persona memory module provides:

- **PersonaType Enum**: Defines different agent archetypes (elderly resident, working parent, young professional, student, chronic patient, etc.)
- **PersonaTemplate**: Detailed agent profiles with demographics, values, beliefs, behavioral tendencies, and healthcare attitudes
- **EmotionalMotivationalState**: Dynamic emotional and motivational state tracking
- **PersonaMemoryManager**: Main interface for managing agent personas and memory systems

### 2. LLM Interaction Layer (`simulation/llm_interaction_layer.py`)

The LLM interaction layer provides:

- **PromptBuilder**: Assembles persona-specific prompts from agent state and observations
- **InferenceRouter**: Routes prompts to appropriate LLM endpoints based on requirements
- **DecisionPlanningEngine**: Orchestrates the agent's cognitive cycle and decision-making process
- **LLMInteractionLayer**: Main interface for LLM-based agent decisions

### 3. Model Integration (`simulation/model.py`)

The model has been enhanced with:

- **LLM Component Initialization**: Automatically initializes LLM components when `needs_selection` or `movement_behavior` is set to "llms"
- **Persona Assignment**: Assigns appropriate personas to residents based on their demographic characteristics
- **Seamless Integration**: Works alongside existing simulation components without breaking compatibility

### 4. Resident Agent Enhancement (`agents/resident.py`)

The resident agent now supports:

- **LLM-based Movement**: Uses LLM decision-making for choosing where to move
- **LLM-based Needs Generation**: Uses persona and emotional state for realistic need generation
- **Emotional State Updates**: Updates emotional state based on experiences (POI visits, waiting times, etc.)
- **Episodic Memory**: Maintains memory of recent experiences for LLM context

## How to Use

### Command Line Arguments

You can enable LLM behavior using the following command line arguments:

```bash
# Enable LLM-based movement behavior
python main.py --movement-behavior llms --residents 10 --steps 100

# Enable LLM-based needs generation
python main.py --needs-selection llms --residents 10 --steps 100

# Enable both LLM behaviors
python main.py --movement-behavior llms --needs-selection llms --residents 10 --steps 100
```

### Example Usage

```bash
# Run a small test simulation with LLM behavior
python main.py --movement-behavior llms --residents 5 --steps 50 --parishes "Taipa" "Coloane"

# Run a comprehensive simulation with both LLM features
python main.py --movement-behavior llms --needs-selection llms --residents 20 --steps 200 --parishes "Taipa" "Coloane"
```

## Technical Details

### Persona Assignment Logic

Personas are assigned based on demographic characteristics:

- **Age 65+**: `ELDERLY_RESIDENT`
- **Age < 25**: `STUDENT` (if student status) or `YOUNG_PROFESSIONAL`
- **Age 25-45**: `WORKING_PARENT` (if family status) or `YOUNG_PROFESSIONAL`
- **Age 45-64**: `WORKING_PARENT` (60% chance) or `YOUNG_PROFESSIONAL`

### LLM Decision Making Process

1. **State Creation**: Convert resident data to structured agent state
2. **Observation Building**: Gather current context (nearby POIs, agents, etc.)
3. **Memory Retrieval**: Get relevant episodic memories
4. **LLM Inference**: Send structured prompt to LLM for decision
5. **Decision Parsing**: Extract actionable decision from LLM response
6. **State Update**: Update emotional state based on decision confidence

### Emotional State Modeling

The system tracks:

- **Current Emotions**: Calm, anxious, frustrated, confident, stressed, hopeful, worried, satisfied
- **Motivational Drives**: Health security, family wellbeing, career advancement, social approval, etc.
- **Dynamic Updates**: Emotions change based on experiences (POI visits, waiting times, decision outcomes)

### Needs Generation

LLM-based needs generation considers:

- **Persona Type**: Different personas have different base needs
- **Emotional State**: Current emotions modify need intensities
- **Stress Level**: High stress increases recreation and healthcare needs
- **Individual Variation**: Random factors ensure agent uniqueness

## Testing

Use the provided test script to verify the integration:

```bash
python test_llm_integration.py
```

This will test:
- Module imports
- Persona manager functionality
- LLM interaction layer initialization

## Architecture Benefits

### 1. Modularity
- LLM components are optional and don't interfere with existing functionality
- Can be enabled/disabled via command line arguments
- Graceful fallback to rule-based behavior if LLM components fail

### 2. Realism
- Persona-driven behavior creates more realistic agent archetypes
- Emotional state modeling adds psychological depth
- Context-aware decision making through LLM integration

### 3. Flexibility
- Multiple persona types support diverse population modeling
- Configurable LLM endpoints for different performance/cost trade-offs
- Extensible framework for adding new persona types or behaviors

### 4. Performance
- Efficient caching of persona templates
- Parallel LLM inference capabilities
- Fallback mechanisms prevent simulation failures

## Configuration

### LLM API Configuration

The system uses the existing LLM API configuration in `config/config_API.py`:

- **ZhipuAI Integration**: Uses the configured ZhipuAI API for LLM inference
- **Model Selection**: Defaults to "glm-4-flash" for fast inference
- **Error Handling**: Graceful fallback if API calls fail

### Persona Customization

You can customize persona templates by modifying the `PersonaTemplateManager` class:

- Add new persona types
- Modify existing persona characteristics
- Adjust emotional state parameters
- Customize need generation logic

## Future Enhancements

Potential areas for future development:

1. **Advanced Memory Systems**: Implement more sophisticated episodic and semantic memory
2. **Social Interactions**: Model agent-to-agent communication and influence
3. **Learning Mechanisms**: Allow agents to learn and adapt over time
4. **Policy Response Modeling**: Simulate how different personas respond to policy changes
5. **Multi-modal Integration**: Incorporate visual and spatial reasoning capabilities

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required modules are installed and paths are correct
2. **LLM API Failures**: Check API configuration and network connectivity
3. **Memory Issues**: Monitor memory usage with large numbers of LLM-enabled agents
4. **Performance**: Consider using faster LLM endpoints for large simulations

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will provide detailed information about:
- Persona assignment process
- LLM decision making steps
- Emotional state updates
- API call details

## Conclusion

The LLM integration transforms the ABM simulation from a rule-based system to a sophisticated, persona-driven model that can capture the complexity of human behavior in healthcare policy scenarios. The modular design ensures compatibility with existing functionality while providing powerful new capabilities for realistic agent modeling. 