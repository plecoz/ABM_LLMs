#!/usr/bin/env python3
"""
LLM Interaction Layer for ABM Simulation

This layer serves as a bridge between the simulation engine and LLM-based agents,
providing context-aware decision making through structured prompts and responses.

Components:
1. Prompt Builder: Assembles persona-specific prompts from agent state and observations
2. Inference Router: Dispatches prompts to appropriate LLM endpoints
3. Decision & Planning Engine: Orchestrates the agent's cognitive cycle
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .llm_api import LLMWrapper
from agents.fifteenminutescity.persona_memory_modules import PersonaMemoryManager, PersonaType


class LLMEndpointType(Enum):
    """Types of LLM endpoints available."""
    HIGH_CAPACITY = "high_capacity"  # For rich reasoning (GPT-4, Claude, etc.)
    FAST_LOCAL = "fast_local"       # For low-latency local models
    FINE_TUNED = "fine_tuned"       # For specialized fine-tuned models


@dataclass
class AgentObservation:
    """Structured observation data for an agent."""
    current_location: str
    nearby_agents: List[Dict[str, Any]]
    nearby_pois: List[Dict[str, Any]]
    environmental_context: Dict[str, Any]
    time_step: int


@dataclass
class AgentState:
    """Current state of an agent."""
    agent_id: str
    role: str
    demographic: Dict[str, Any]
    current_needs: Dict[str, float]
    utilities: List[str]
    sdg_constraints: List[str]
    location: str
    energy_level: float
    current_activity: Optional[str]


@dataclass
class EpisodicMemory:
    """Represents a memory from the agent's past experiences."""
    timestamp: int
    location: str
    action: str
    outcome: str
    satisfaction_gained: float
    other_agents_involved: List[str]


@dataclass
class LLMDecision:
    """Structured decision output from LLM."""
    action: str
    rationale: str
    confidence: float
    alternative_actions: List[str]
    expected_utility: float


class PromptBuilder:
    """Assembles persona-specific prompts from agent state and observations."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build_prompt(
        self, 
        agent_state: AgentState, 
        observation: AgentObservation,
        episodic_memories: List[EpisodicMemory],
        top_k_memories: int = 3
    ) -> str:
        """
        Build a structured prompt for the LLM based on agent state and context.
        
        Args:
            agent_state: Current state of the agent
            observation: Current observations
            episodic_memories: Agent's past experiences
            top_k_memories: Number of most relevant memories to include
            
        Returns:
            Formatted prompt string
        """
        # Select top-k most relevant memories (simplified - by recency for now)
        relevant_memories = self._select_relevant_memories(
            episodic_memories, observation, top_k_memories
        )
        
        # Build the structured prompt
        prompt = self._construct_prompt_template(
            agent_state, observation, relevant_memories
        )
        
        return prompt
    
    def _select_relevant_memories(
        self, 
        memories: List[EpisodicMemory], 
        observation: AgentObservation,
        top_k: int
    ) -> List[EpisodicMemory]:
        """Select the most relevant memories for the current context."""
        if not memories:
            return []
        
        # For now, use simple recency-based selection
        # Later can be enhanced with semantic similarity
        sorted_memories = sorted(memories, key=lambda m: m.timestamp, reverse=True)
        return sorted_memories[:top_k]
    
    def _construct_prompt_template(
        self,
        agent_state: AgentState,
        observation: AgentObservation,
        memories: List[EpisodicMemory]
    ) -> str:
        """Construct the actual prompt template."""
        
        # Format demographic info
        demographic_str = ", ".join([f"{k}: {v}" for k, v in agent_state.demographic.items()])
        
        # Format utilities and SDG constraints
        utilities_str = ", ".join(agent_state.utilities) if agent_state.utilities else "general well-being"
        sdg_str = ", ".join(agent_state.sdg_constraints) if agent_state.sdg_constraints else "sustainable development"
        
        # Format current needs
        needs_str = ", ".join([f"{need}: {level:.1f}" for need, level in agent_state.current_needs.items()])
        
        # Format memories
        memory_str = ""
        if memories:
            memory_entries = []
            for mem in memories:
                memory_entries.append(
                    f"Time {mem.timestamp}: At {mem.location}, performed '{mem.action}' -> {mem.outcome} "
                    f"(satisfaction: {mem.satisfaction_gained:.1f})"
                )
            memory_str = "\n".join(memory_entries)
        else:
            memory_str = "No relevant past experiences."
        
        # Format nearby POIs
        poi_str = ""
        if observation.nearby_pois:
            poi_entries = [f"- {poi['name']} ({poi['type']})" for poi in observation.nearby_pois]
            poi_str = "\n".join(poi_entries)
        else:
            poi_str = "No nearby points of interest."
        
        # Format nearby agents
        agents_str = ""
        if observation.nearby_agents:
            agent_entries = [f"- Agent {agent['id']} ({agent.get('role', 'resident')})" 
                           for agent in observation.nearby_agents]
            agents_str = "\n".join(agent_entries)
        else:
            agents_str = "No other agents nearby."
        
        # Construct the full prompt
        prompt = f"""System: You are {agent_state.role}, representing {demographic_str}. 
Goal: Maximize {utilities_str} while adhering to {sdg_str} constraints.

Current Status:
- Location: {agent_state.location}
- Energy Level: {agent_state.energy_level:.1f}
- Current Activity: {agent_state.current_activity or 'None'}
- Current Needs: {needs_str}

Memory (Recent Experiences):
{memory_str}

Current Observation:
- Time Step: {observation.time_step}
- Nearby Points of Interest:
{poi_str}
- Nearby Agents:
{agents_str}

Task: Based on your current needs, past experiences, and observations, decide on ONE specific action to take next. Provide a brief rationale for your choice.

Response Format:
Action: [specific action]
Rationale: [brief explanation]
Confidence: [0.0-1.0]
Expected_Utility: [0.0-1.0]"""

        return prompt


class InferenceRouter:
    """Routes prompts to appropriate LLM endpoints based on requirements."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_endpoints = {}
        self._initialize_endpoints()
    
    def _initialize_endpoints(self):
        """Initialize available LLM endpoints."""
        # High-capacity models for rich reasoning
        try:
            self.llm_endpoints[LLMEndpointType.HIGH_CAPACITY] = LLMWrapper(
                model_name="gpt4omini",  # Fast and capable
                platform="OpenAI"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize high-capacity endpoint: {e}")
        
        # Fast local models for low-latency responses
        try:
            self.llm_endpoints[LLMEndpointType.FAST_LOCAL] = LLMWrapper(
                model_name="llama3-8b",
                platform="SiliconFlow"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize fast local endpoint: {e}")
        
        # Log available endpoints
        available = list(self.llm_endpoints.keys())
        self.logger.info(f"Initialized LLM endpoints: {available}")
    
    def route_request(
        self, 
        prompt: str, 
        agent_complexity: str = "standard",
        latency_requirement: str = "normal"
    ) -> Tuple[LLMEndpointType, str]:
        """
        Route a prompt to the appropriate LLM endpoint.
        
        Args:
            prompt: The prompt to send
            agent_complexity: "simple" | "standard" | "complex"
            latency_requirement: "low" | "normal" | "high"
            
        Returns:
            Tuple of (endpoint_type, response)
        """
        # Simple routing logic (can be enhanced later)
        if latency_requirement == "low" and LLMEndpointType.FAST_LOCAL in self.llm_endpoints:
            endpoint_type = LLMEndpointType.FAST_LOCAL
        elif agent_complexity == "complex" and LLMEndpointType.HIGH_CAPACITY in self.llm_endpoints:
            endpoint_type = LLMEndpointType.HIGH_CAPACITY
        else:
            # Default to first available endpoint
            endpoint_type = next(iter(self.llm_endpoints.keys()))
        
        try:
            response = self.llm_endpoints[endpoint_type].get_response(prompt)
            self.logger.debug(f"Routed to {endpoint_type.value}, got response length: {len(response)}")
            return endpoint_type, response
        except Exception as e:
            self.logger.error(f"Error with endpoint {endpoint_type.value}: {e}")
            # Fallback to any available endpoint
            for fallback_type, llm in self.llm_endpoints.items():
                if fallback_type != endpoint_type:
                    try:
                        response = llm.get_response(prompt)
                        self.logger.info(f"Fallback to {fallback_type.value} successful")
                        return fallback_type, response
                    except Exception as fallback_e:
                        self.logger.error(f"Fallback {fallback_type.value} also failed: {fallback_e}")
            
            # If all endpoints fail, return a default response
            return endpoint_type, "Action: wait\nRationale: System error, waiting for next step\nConfidence: 0.1\nExpected_Utility: 0.0"


class DecisionPlanningEngine:
    """Orchestrates the agent's cognitive cycle and decision-making process."""
    
    def __init__(self):
        self.prompt_builder = PromptBuilder()
        self.inference_router = InferenceRouter()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def make_decision(
        self,
        agent_state: AgentState,
        observation: AgentObservation,
        episodic_memories: List[EpisodicMemory],
        agent_complexity: str = "standard",
        latency_requirement: str = "normal"
    ) -> LLMDecision:
        """
        Execute the full decision-making cycle for an agent.
        
        Args:
            agent_state: Current state of the agent
            observation: Current observations
            episodic_memories: Agent's memory buffer
            agent_complexity: Complexity level for routing
            latency_requirement: Latency requirement for routing
            
        Returns:
            Structured decision object
        """
        try:
            # Step 1: Build the prompt
            prompt = self.prompt_builder.build_prompt(
                agent_state, observation, episodic_memories
            )
            
            # Step 2: Route to appropriate LLM
            endpoint_type, raw_response = self.inference_router.route_request(
                prompt, agent_complexity, latency_requirement
            )
            
            # Step 3: Parse the response
            decision = self._parse_llm_response(raw_response)
            
            self.logger.debug(f"Decision made for agent {agent_state.agent_id}: {decision.action}")
            return decision
            
        except Exception as e:
            self.logger.error(f"Error in decision making for agent {agent_state.agent_id}: {e}")
            # Return a safe default decision
            return LLMDecision(
                action="wait",
                rationale="Error in decision process, waiting for next step",
                confidence=0.1,
                alternative_actions=["go_home"],
                expected_utility=0.0
            )
    
    def _parse_llm_response(self, raw_response: str) -> LLMDecision:
        """Parse the raw LLM response into a structured decision."""
        try:
            # Initialize default values
            action = "wait"
            rationale = "No clear decision provided"
            confidence = 0.5
            expected_utility = 0.5
            alternative_actions = []
            
            # Parse the response line by line
            lines = raw_response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Action:'):
                    action = line.replace('Action:', '').strip()
                elif line.startswith('Rationale:'):
                    rationale = line.replace('Rationale:', '').strip()
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.replace('Confidence:', '').strip())
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                    except ValueError:
                        confidence = 0.5
                elif line.startswith('Expected_Utility:'):
                    try:
                        expected_utility = float(line.replace('Expected_Utility:', '').strip())
                        expected_utility = max(0.0, min(1.0, expected_utility))  # Clamp to [0,1]
                    except ValueError:
                        expected_utility = 0.5
            
            return LLMDecision(
                action=action,
                rationale=rationale,
                confidence=confidence,
                alternative_actions=alternative_actions,
                expected_utility=expected_utility
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return LLMDecision(
                action="wait",
                rationale="Failed to parse LLM response",
                confidence=0.1,
                alternative_actions=[],
                expected_utility=0.0
            )


class PathScoringRequest:
    """Request for LLM to score multiple path options"""
    def __init__(self, agent_id, path_options, context):
        self.agent_id = agent_id
        self.path_options = path_options  # List of path dictionaries with characteristics
        self.context = context  # Agent context (personality, current needs, time, etc.)

class PathScoringResponse:
    """Response from LLM path scoring"""
    def __init__(self, selected_path_id, reasoning, confidence):
        self.selected_path_id = selected_path_id
        self.reasoning = reasoning
        self.confidence = confidence

class LLMInteractionLayer:
    """Main interface for the LLM Interaction Layer."""
    
    def __init__(self):
        self.decision_engine = DecisionPlanningEngine()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("LLM Interaction Layer initialized")
        self.llm_client = None  # Placeholder for actual LLM client
        self.path_scoring_enabled = True
    
    def get_agent_decision(
        self,
        agent_state: AgentState,
        observation: AgentObservation,
        episodic_memories: List[EpisodicMemory] = None,
        agent_complexity: str = "standard",
        latency_requirement: str = "normal"
    ) -> LLMDecision:
        """
        Main entry point for getting LLM-based decisions for agents.
        
        Args:
            agent_state: Current state of the agent
            observation: Current observations
            episodic_memories: Agent's memory buffer (optional)
            agent_complexity: Complexity level for LLM routing
            latency_requirement: Latency requirement for LLM routing
            
        Returns:
            Structured decision from LLM
        """
        if episodic_memories is None:
            episodic_memories = []
        
        return self.decision_engine.make_decision(
            agent_state=agent_state,
            observation=observation,
            episodic_memories=episodic_memories,
            agent_complexity=agent_complexity,
            latency_requirement=latency_requirement
        )
    
    def create_agent_state_from_resident(self, resident_agent) -> AgentState:
        """
        Helper method to create AgentState from a Resident agent.
        
        Args:
            resident_agent: A Resident agent instance
            
        Returns:
            AgentState object
        """
        # Extract demographic information
        demographic = {
            'age': getattr(resident_agent, 'age', 30),
            'gender': getattr(resident_agent, 'gender', 'unspecified'),
            'education': getattr(resident_agent, 'education', 'high_school'),
            'employment_status': getattr(resident_agent, 'employment_status', 'unknown'),
            'parish': getattr(resident_agent, 'parish', 'unknown')
        }
        
        # Extract current needs
        current_needs = getattr(resident_agent, 'current_needs', {})
        if not current_needs:
            current_needs = getattr(resident_agent, 'dynamic_needs', {})
        
        # Determine utilities based on agent characteristics
        utilities = self._determine_utilities(resident_agent)
        
        # Determine SDG constraints (simplified for now)
        sdg_constraints = ["sustainable cities", "good health", "quality education"]
        
        return AgentState(
            agent_id=str(resident_agent.unique_id),
            role="resident",
            demographic=demographic,
            current_needs=current_needs,
            utilities=utilities,
            sdg_constraints=sdg_constraints,
            location=getattr(resident_agent, 'current_node', 'unknown'),
            energy_level=getattr(resident_agent, 'energy_level', 1.0),
            current_activity=getattr(resident_agent, 'current_activity', None)
        )
    
    def create_observation_from_context(self, agent, model) -> AgentObservation:
        """
        Helper method to create AgentObservation from simulation context.
        
        Args:
            agent: The agent making the observation
            model: The simulation model
            
        Returns:
            AgentObservation object
        """
        # Get nearby agents (simplified)
        nearby_agents = []
        if hasattr(model, 'get_nearby_agents'):
            nearby_agents = [
                {
                    'id': str(other_agent.unique_id),
                    'role': 'resident',
                    'location': getattr(other_agent, 'current_node', 'unknown')
                }
                for other_agent in model.get_nearby_agents(agent)
            ]
        
        # Get nearby POIs (simplified)
        nearby_pois = []
        if hasattr(agent, 'accessible_nodes') and agent.accessible_nodes:
            for poi_type, poi_list in agent.accessible_nodes.items():
                if poi_type != 'all_nodes':  # Skip the 'all_nodes' key
                    for poi in poi_list[:3]:  # Limit to 3 POIs per type
                        nearby_pois.append({
                            'name': f"{poi_type}_{poi}",
                            'type': poi_type,
                            'id': poi
                        })
        
        return AgentObservation(
            current_location=str(getattr(agent, 'current_node', 'unknown')),
            nearby_agents=nearby_agents,
            nearby_pois=nearby_pois,
            environmental_context={'weather': 'normal', 'time_of_day': 'day'},
            time_step=getattr(model, 'step_count', 0)
        )
    
    def _determine_utilities(self, agent) -> List[str]:
        """Determine utilities based on agent characteristics."""
        utilities = ["well-being", "convenience"]
        
        # Add age-specific utilities
        age = getattr(agent, 'age', 30)
        if age < 25:
            utilities.extend(["education", "social_interaction"])
        elif age > 65:
            utilities.extend(["health", "comfort"])
        else:
            utilities.extend(["work_efficiency", "family_time"])
        
        return utilities
    
    def score_path_options(self, agent_state, path_options, context):
        """
        Use LLM to score and select the best path from multiple options.
        This is the single implementation for path scoring - no duplication.
        
        Args:
            agent_state: Current state of the agent (dict with age, needs, etc.)
            path_options: List of path dictionaries with OSM metadata
            context: Additional context (time_of_day, etc.)
            
        Returns:
            PathScoringResponse with selected path and reasoning
        """
        if not path_options:
            return PathScoringResponse(0, "No paths available", 0.0)
        
        try:
            # Create LLM prompt
            prompt = self._create_path_scoring_prompt(agent_state, path_options, context)
            
            # Call LLM (placeholder for actual implementation)
            selected_path_id = self._call_llm_api(prompt, path_options)
            
            # Return response
            return PathScoringResponse(
                selected_path_id=selected_path_id,
                reasoning=f"LLM selected path {selected_path_id + 1} based on efficiency and safety",
                confidence=0.8
            )
            
        except Exception as e:
            # Fallback to rule-based selection
            return self._fallback_selection(path_options)

    def _create_path_scoring_prompt(self, agent_state, path_options, context):
        """
        Create LLM prompt for path scoring (1-10 scale).
        
        Args:
            agent_state: Agent characteristics
            path_options: Available paths with OSM metadata
            context: Decision context
            
        Returns:
            Formatted prompt string
        """
        age = agent_state.get('age', 30)
        time_of_day = context.get('time_of_day', 12)
        
        prompt = f"""You are helping a {age}-year-old resident choose the best path at {time_of_day}:00.

Score each path from 1-10 (10 = best choice) considering:
- Travel time efficiency
- Road safety and comfort
- Road type appropriateness
- Green area coverage (parks, forests - higher is more pleasant)
- Overall convenience

Available paths:
"""
        
        for path in path_options:
            green_desc = f"{path.get('green_area_percentage', 0)}% through green areas"
            prompt += f"""
Path {path['path_id']}:
- Distance: {path['distance_meters']} meters
- Travel time: {path['travel_time_minutes']} minutes
- Main road type: {path['dominant_road_type']}
- Road types: {', '.join(path['road_types'][:3])}
- Segments: {path['total_segments']}
- Green coverage: {green_desc}
"""
        
        prompt += f"""
Respond with ONLY the path number (1-{len(path_options)}) that has the highest score.
Example: 2
"""
        
        return prompt

    def _call_llm_api(self, prompt, path_options):
        """
        Call LLM API for path scoring.
        
        Args:
            prompt: Formatted prompt
            path_options: Available path options
            
        Returns:
            Index of selected path (0-based)
        """
        # PLACEHOLDER: Replace with actual LLM API call
        # 
        # Example integration:
        # response = your_llm_service.generate(prompt)
        # selected_path_number = int(response.strip())
        # return selected_path_number - 1  # Convert to 0-based index
        
        # For now, use rule-based fallback
        return self._rule_based_scoring(path_options)

    def _rule_based_scoring(self, path_options):
        """
        Rule-based scoring when LLM is not available.
        Now includes green area coverage in scoring.
        
        Args:
            path_options: Available path options
            
        Returns:
            Index of best path (0-based)
        """
        scores = []
        
        for path in path_options:
            score = 0
            
            # Time efficiency (50% weight)
            min_time = min(p['travel_time_minutes'] for p in path_options)
            max_time = max(p['travel_time_minutes'] for p in path_options)
            if max_time > min_time:
                time_score = 1.0 - ((path['travel_time_minutes'] - min_time) / (max_time - min_time))
            else:
                time_score = 1.0
            score += 5.0 * time_score
            
            # Road type safety (30% weight)
            road_type_scores = {
                'residential': 4.0, 'tertiary': 3.5, 'secondary': 3.0,
                'primary': 2.5, 'trunk': 2.0, 'motorway': 1.5,
                'footway': 4.5, 'path': 4.0, 'unclassified': 2.5
            }
            road_score = road_type_scores.get(path['dominant_road_type'], 2.5)
            score += 3.0 * (road_score / 5.0)
            
            # Green area coverage (20% weight)
            green_percentage = path.get('green_area_percentage', 0)
            green_score = min(1.0, green_percentage / 50.0)  # Normalize to 0-1 (50% green = max score)
            score += 2.0 * green_score
            
            scores.append(score)
        
        # Return index of highest scoring path
        return scores.index(max(scores))

    def _fallback_selection(self, path_options):
        """
        Simple fallback path selection.
        
        Args:
            path_options: Available path options
            
        Returns:
            PathScoringResponse
        """
        # Select shortest time path as fallback
        shortest_idx = min(range(len(path_options)), 
                         key=lambda i: path_options[i]['travel_time_minutes'])
        
        return PathScoringResponse(
            selected_path_id=shortest_idx,
            reasoning="Selected shortest time path (fallback)",
            confidence=0.5
        ) 