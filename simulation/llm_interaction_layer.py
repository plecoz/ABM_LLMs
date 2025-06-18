#!/usr/bin/env python3
"""
General LLM Interaction Layer for ABM Simulations

This layer serves as a bridge between simulation engines and LLM-based agents,
providing context-aware decision making through structured prompts and responses.
This is a general-purpose layer that can be used across different types of simulations
(healthcare, climate change, immigration, poverty, etc.).

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
    """Structured observation data for an agent - general purpose."""
    current_location: str
    nearby_agents: List[Dict[str, Any]]
    nearby_entities: List[Dict[str, Any]]  # Generic entities (POIs, resources, etc.)
    environmental_context: Dict[str, Any]
    time_step: int


@dataclass
class AgentState:
    """Current state of an agent - general purpose."""
    agent_id: str
    agent_type: str  # More general than 'role'
    demographic: Dict[str, Any]
    current_needs: Dict[str, float]
    utilities: List[str]
    constraints: List[str]  # More general than 'sdg_constraints'
    location: str
    energy_level: float
    current_activity: Optional[str]


@dataclass
class EpisodicMemory:
    """Represents a memory from the agent's past experiences - general purpose."""
    timestamp: int
    location: str
    action: str
    outcome: str
    satisfaction_gained: float
    other_agents_involved: List[str]


@dataclass
class LLMDecision:
    """Structured decision output from LLM - general purpose."""
    action: str
    rationale: str
    confidence: float
    alternative_actions: List[str]
    expected_utility: float


class PromptBuilder:
    """Assembles persona-specific prompts from agent state and observations - general purpose."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build_prompt(
        self, 
        agent_state: AgentState, 
        observation: AgentObservation,
        episodic_memories: List[EpisodicMemory],
        top_k_memories: int = 3,
        simulation_context: Dict[str, Any] = None
    ) -> str:
        """
        Build a structured prompt for the LLM based on agent state and context.
        
        Args:
            agent_state: Current state of the agent
            observation: Current observations
            episodic_memories: Agent's past experiences
            top_k_memories: Number of most relevant memories to include
            simulation_context: Additional context specific to the simulation type
            
        Returns:
            Formatted prompt string
        """
        # Select top-k most relevant memories (simplified - by recency for now)
        relevant_memories = self._select_relevant_memories(
            episodic_memories, observation, top_k_memories
        )
        
        # Build the structured prompt
        prompt = self._construct_prompt_template(
            agent_state, observation, relevant_memories, simulation_context
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
        memories: List[EpisodicMemory],
        simulation_context: Dict[str, Any] = None
    ) -> str:
        """Construct the actual prompt template - general purpose."""
        
        # Format demographic info
        demographic_str = ", ".join([f"{k}: {v}" for k, v in agent_state.demographic.items()])
        
        # Format utilities and constraints
        utilities_str = ", ".join(agent_state.utilities) if agent_state.utilities else "general well-being"
        constraints_str = ", ".join(agent_state.constraints) if agent_state.constraints else "none specified"
        
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
        
        # Format nearby entities (generic)
        entities_str = ""
        if observation.nearby_entities:
            entity_entries = [f"- {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})" 
                            for entity in observation.nearby_entities]
            entities_str = "\n".join(entity_entries)
        else:
            entities_str = "No nearby entities of interest."
        
        # Format nearby agents
        agents_str = ""
        if observation.nearby_agents:
            agent_entries = [f"- Agent {agent['id']} ({agent.get('type', 'unknown')})" 
                           for agent in observation.nearby_agents]
            agents_str = "\n".join(agent_entries)
        else:
            agents_str = "No other agents nearby."
        
        # Add simulation-specific context if provided
        context_str = ""
        if simulation_context:
            context_entries = [f"- {k}: {v}" for k, v in simulation_context.items()]
            context_str = f"\n\nSimulation Context:\n" + "\n".join(context_entries)
        
        # Construct the full prompt
        prompt = f"""System: You are a {agent_state.agent_type}, representing {demographic_str}. 
Goal: Maximize {utilities_str} while adhering to these constraints: {constraints_str}.

Current Status:
- Location: {agent_state.location}
- Energy Level: {agent_state.energy_level:.1f}
- Current Activity: {agent_state.current_activity or 'None'}
- Current Needs: {needs_str}

Memory (Recent Experiences):
{memory_str}

Current Observation:
- Time Step: {observation.time_step}
- Nearby Entities:
{entities_str}
- Nearby Agents:
{agents_str}{context_str}

Task: Based on your current needs, past experiences, and observations, decide on ONE specific action to take next. Provide a brief rationale for your choice.

Response Format:
Action: [specific action]
Rationale: [brief explanation]
Confidence: [0.0-1.0]
Expected_Utility: [0.0-1.0]"""

        return prompt


class InferenceRouter:
    """Routes prompts to appropriate LLM endpoints based on requirements - general purpose."""
    
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
    """Orchestrates the agent's cognitive cycle and decision-making process - general purpose."""
    
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
        latency_requirement: str = "normal",
        simulation_context: Dict[str, Any] = None
    ) -> LLMDecision:
        """
        Execute the full decision-making cycle for an agent.
        
        Args:
            agent_state: Current state of the agent
            observation: Current observations
            episodic_memories: Agent's memory buffer
            agent_complexity: Complexity level for routing
            latency_requirement: Latency requirement for routing
            simulation_context: Additional context specific to the simulation type
            
        Returns:
            Structured decision object
        """
        try:
            # Step 1: Build the prompt
            prompt = self.prompt_builder.build_prompt(
                agent_state, observation, episodic_memories, simulation_context=simulation_context
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
                alternative_actions=["continue_current_activity"],
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


class LLMInteractionLayer:
    """Main interface for the general LLM Interaction Layer."""
    
    def __init__(self):
        self.decision_engine = DecisionPlanningEngine()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("General LLM Interaction Layer initialized")
    
    def get_agent_decision(
        self,
        agent_state: AgentState,
        observation: AgentObservation,
        episodic_memories: List[EpisodicMemory] = None,
        agent_complexity: str = "standard",
        latency_requirement: str = "normal",
        simulation_context: Dict[str, Any] = None
    ) -> LLMDecision:
        """
        Main entry point for getting LLM-based decisions for agents.
        
        Args:
            agent_state: Current state of the agent
            observation: Current observations
            episodic_memories: Agent's memory buffer (optional)
            agent_complexity: Complexity level for LLM routing
            latency_requirement: Latency requirement for LLM routing
            simulation_context: Additional context specific to the simulation type
            
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
            latency_requirement=latency_requirement,
            simulation_context=simulation_context
        ) 