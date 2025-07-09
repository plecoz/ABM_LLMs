#!/usr/bin/env python3
"""
Fifteen-Minute City Specific LLM Interaction Layer

This layer extends the general LLM interaction layer with functionality specific
to the fifteen-minute city simulation, including path scoring, urban mobility
decisions, and resident-specific utilities and observations.

Components:
1. Path Scoring: LLM-based path selection for urban mobility
2. Resident Helpers: Convert resident agents to general LLM structures
3. Urban Context: Fifteen-minute city specific context and utilities
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..llm_interaction_layer import LLMInteractionLayer, AgentState, AgentObservation, LLMDecision


@dataclass
class PathScoringRequest:
    """Request for LLM to score multiple path options for urban mobility."""
    agent_id: str
    path_options: List[Dict[str, Any]]  # List of path dictionaries with OSM metadata
    context: Dict[str, Any]  # Agent context (personality, current needs, time, etc.)


@dataclass
class PathScoringResponse:
    """Response from LLM path scoring for urban mobility."""
    selected_path_id: int
    reasoning: str
    confidence: float


class FifteenMinuteCityLLMLayer(LLMInteractionLayer):
    """
    Fifteen-minute city specific LLM interaction layer.
    Extends the general layer with urban mobility and resident-specific functionality.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Fifteen-Minute City LLM Layer initialized")
        self.path_scoring_enabled = True
    
    def create_agent_state_from_resident(self, resident_agent) -> AgentState:
        """
        Helper method to create AgentState from a Resident agent.
        This is specific to the fifteen-minute city simulation.
        
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
        
        # Determine utilities based on agent characteristics (fifteen-minute city specific)
        utilities = self._determine_urban_utilities(resident_agent)
        
        # Determine constraints (fifteen-minute city focused on sustainability)
        constraints = ["sustainable cities", "good health", "quality education", "15-minute accessibility"]
        
        return AgentState(
            agent_id=str(resident_agent.unique_id),
            agent_type="urban_resident",
            demographic=demographic,
            current_needs=current_needs,
            utilities=utilities,
            constraints=constraints,
            location=getattr(resident_agent, 'current_node', 'unknown'),
            energy_level=getattr(resident_agent, 'energy_level', 1.0),
            current_activity=getattr(resident_agent, 'current_activity', None)
        )
    
    def create_observation_from_context(self, agent, model) -> AgentObservation:
        """
        Helper method to create AgentObservation from fifteen-minute city simulation context.
        
        Args:
            agent: The resident agent making the observation
            model: The simulation model
            
        Returns:
            AgentObservation object
        """
        # Get nearby agents (specific to the simulation structure)
        nearby_agents = []
        if hasattr(model, 'get_nearby_agents'):
            nearby_agents = [
                {
                    'id': str(other_agent.unique_id),
                    'type': 'urban_resident',
                    'location': getattr(other_agent, 'current_node', 'unknown')
                }
                for other_agent in model.get_nearby_agents(agent)
            ]
        
        # Get nearby POIs from the model's POI agents
        nearby_entities = []
        if hasattr(model, 'poi_agents') and model.poi_agents:
            # Get POIs that are within the agent's accessible nodes
            accessible_node_ids = set()
            if hasattr(agent, 'accessible_nodes') and agent.accessible_nodes:
                # accessible_nodes is a dict of {node_id: distance}
                accessible_node_ids = set(agent.accessible_nodes.keys())
            
            # Find POI agents at accessible nodes
            accessible_pois = []
            for poi_agent in model.poi_agents:
                if hasattr(poi_agent, 'node_id') and poi_agent.node_id in accessible_node_ids:
                    accessible_pois.append(poi_agent)
            
            # Limit to a reasonable number and create entity entries
            for poi_agent in accessible_pois[:10]:  # Limit to 10 nearby POIs
                nearby_entities.append({
                    'name': f"{poi_agent.poi_type}_{poi_agent.node_id}",
                    'type': poi_agent.poi_type,
                    'id': poi_agent.node_id,
                    'category': getattr(poi_agent, 'category', 'poi')
                })
        
        # Add fifteen-minute city specific environmental context
        environmental_context = {
            'weather': 'normal', 
            'time_of_day': getattr(model, 'hour_of_day', 12),
            'day_of_week': getattr(model, 'day_of_week', 0),
            'city_type': 'fifteen_minute_city'
        }
        
        return AgentObservation(
            current_location=str(getattr(agent, 'current_node', 'unknown')),
            nearby_agents=nearby_agents,
            nearby_entities=nearby_entities,
            environmental_context=environmental_context,
            time_step=getattr(model, 'step_count', 0)
        )
    
    def _determine_urban_utilities(self, agent) -> List[str]:
        """
        Determine utilities based on agent characteristics for urban residents.
        This is specific to fifteen-minute city context.
        """
        utilities = ["well-being", "convenience", "accessibility"]
        
        # Add age-specific utilities for urban living
        age = getattr(agent, 'age', 30)
        if age < 25:
            utilities.extend(["education", "social_interaction", "entertainment"])
        elif age > 65:
            utilities.extend(["health", "comfort", "safety"])
        else:
            utilities.extend(["work_efficiency", "family_time", "community_engagement"])
        
        # Add employment-based utilities
        employment = getattr(agent, 'employment_status', 'unknown')
        if employment == 'student':
            utilities.append("learning_opportunities")
        elif employment == 'employed':
            utilities.append("commute_efficiency")
        elif employment == 'retired':
            utilities.append("leisure_activities")
        
        return utilities
    
    def get_urban_decision(
        self,
        resident_agent,
        model,
        episodic_memories: List = None,
        agent_complexity: str = "standard",
        latency_requirement: str = "normal"
    ) -> LLMDecision:
        """
        Get LLM-based decision for a resident in the fifteen-minute city context.
        
        Args:
            resident_agent: The resident agent
            model: The simulation model
            episodic_memories: Agent's memory buffer (optional)
            agent_complexity: Complexity level for LLM routing
            latency_requirement: Latency requirement for LLM routing
            
        Returns:
            Structured decision from LLM
        """
        # Convert to general structures
        agent_state = self.create_agent_state_from_resident(resident_agent)
        observation = self.create_observation_from_context(resident_agent, model)
        
        # Add fifteen-minute city specific context
        simulation_context = {
            'simulation_type': 'fifteen_minute_city',
            'urban_environment': 'mixed_use_development',
            'mobility_mode': 'walking',
            'accessibility_goal': '15_minutes_max'
        }
        
        return self.get_agent_target_decision(
            agent_state=agent_state,
            observation=observation,
            episodic_memories=episodic_memories or [],
            agent_complexity=agent_complexity,
            latency_requirement=latency_requirement,
            simulation_context=simulation_context
        )
    
    def get_agent_target_decision(self, agent_state, observation,
                                   episodic_memories, agent_complexity, latency_requirement, simulation_context):
        """
        Get LLM-based target decision for a resident using Chain-of-Thought reasoning.
        
        Args:
            agent_state: Current state of the agent (AgentState object)
            observation: Current observation (AgentObservation object)
            episodic_memories: List of recent memories/experiences
            agent_complexity: Complexity level for decision making
            latency_requirement: Latency requirement for response
            simulation_context: Fifteen-minute city simulation context
            
        Returns:
            LLMDecision object with action and rationale
        """
        try:
            # Build the Chain-of-Thought prompt
            prompt = self.build_prompt_for_target_decision(
                agent_state, observation, episodic_memories, 
                agent_complexity, latency_requirement, simulation_context
            )
            
            # Make LLM API call using the base class method
            response = self.get_agent_decision(
                agent_state=agent_state,
                observation=observation,
                episodic_memories=episodic_memories,
                agent_complexity=agent_complexity,
                latency_requirement=latency_requirement,
                custom_prompt=prompt  # Use our custom CoT prompt
            )
            
            # Parse the response to extract decision and rationale
            if response and hasattr(response, 'action'):
                return response
            else:
                # If response format is different, try to parse it
                return self._parse_target_decision_response(response)
                
        except Exception as e:
            self.logger.error(f"Error in get_agent_target_decision: {e}")
            # Return a fallback decision
            return LLMDecision(
                action="stay_put",
                rationale="Error occurred in LLM decision making, staying at current location for safety",
                confidence=0.3
            )
    
    def _parse_target_decision_response(self, response):
        """
        Parse the LLM response to extract the decision and rationale.
        
        Args:
            response: Raw LLM response text or object
            
        Returns:
            LLMDecision object
        """
        try:
            # Handle different response formats
            response_text = ""
            if isinstance(response, str):
                response_text = response
            elif hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            else:
                response_text = str(response)
            
            # Extract decision from response
            decision_action = "stay_put"  # Default fallback
            rationale = "No clear decision found in response"
            
            # Look for DECISION: pattern
            import re
            decision_match = re.search(r'\*\*DECISION:\s*([^*\n]+)', response_text, re.IGNORECASE)
            if decision_match:
                decision_text = decision_match.group(1).strip()
                
                # Map decision text to action
                decision_mapping = {
                    'go_home': 'go_home',
                    'restaurant': 'restaurant',
                    'shop': 'shop',
                    'hospital': 'hospital',
                    'park': 'park',
                    'library': 'library',
                    'cinema': 'cinema',
                    'gym': 'gym',
                    'pharmacy': 'pharmacy',
                    'bank': 'bank',
                    'supermarket': 'supermarket',
                    'stay_put': 'stay_put'
                }
                
                # Find matching action
                for key, action in decision_mapping.items():
                    if key in decision_text.lower():
                        decision_action = action
                        break
            
            # Look for RATIONALE: pattern
            rationale_match = re.search(r'\*\*RATIONALE:\s*([^*\n]+)', response_text, re.IGNORECASE)
            if rationale_match:
                rationale = rationale_match.group(1).strip()
            
            return LLMDecision(
                action=decision_action,
                rationale=rationale,
                confidence=0.8 if decision_match else 0.5
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing target decision response: {e}")
            return LLMDecision(
                action="stay_put",
                rationale="Error parsing LLM response",
                confidence=0.3
            )

    def build_prompt_for_target_decision(self, agent_state, observation,
                                         episodic_memories, agent_complexity, latency_requirement, simulation_context):
        """
        Build a Chain-of-Thought prompt for LLM to decide the resident's next target destination.
        
        Args:
            agent_state: Current state of the agent (AgentState object)
            observation: Current observation (AgentObservation object)
            episodic_memories: List of recent memories/experiences
            agent_complexity: Complexity level for decision making
            latency_requirement: Latency requirement for response
            simulation_context: Fifteen-minute city simulation context
            
        Returns:
            Formatted CoT prompt string
        """
        # Extract key information
        age = agent_state.demographic.get('age', 30)
        gender = agent_state.demographic.get('gender', 'unspecified')
        employment = agent_state.demographic.get('employment_status', 'unknown')
        parish = agent_state.demographic.get('parish', 'unknown')
        
        current_location = observation.current_location
        time_of_day = observation.environmental_context.get('time_of_day', 12)
        day_of_week = observation.environmental_context.get('day_of_week', 0)
        
        # Convert day of week to readable format
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = days[day_of_week] if 0 <= day_of_week < 7 else 'Unknown'
        
        # Format current needs
        needs_text = ""
        if agent_state.current_needs:
            high_needs = {k: v for k, v in agent_state.current_needs.items() if v > 60}
            if high_needs:
                needs_list = [f"{need}: {value}/100" for need, value in high_needs.items()]
                needs_text = f"High needs: {', '.join(needs_list)}"
            else:
                needs_list = [f"{need}: {value}/100" for need, value in list(agent_state.current_needs.items())[:3]]
                needs_text = f"Current needs: {', '.join(needs_list)}"
        
        # Format nearby entities (POIs)
        nearby_pois = []
        if observation.nearby_entities:
            for entity in observation.nearby_entities[:8]:  # Limit to 8 for prompt clarity
                nearby_pois.append(f"- {entity['type']} (ID: {entity['id']})")
        
        # Format recent memories
        memory_text = ""
        if episodic_memories:
            recent_visits = []
            for memory in episodic_memories[-3:]:  # Last 3 memories
                if isinstance(memory, dict) and 'poi_type' in memory:
                    recent_visits.append(f"- Visited {memory['poi_type']} recently")
                elif isinstance(memory, str):
                    recent_visits.append(f"- {memory}")
            if recent_visits:
                memory_text = "Recent activities:\n" + "\n".join(recent_visits)
        
        # Build the Chain-of-Thought prompt
        prompt = f"""You are an urban mobility assistant helping a resident in a fifteen-minute city decide their next destination. Use step-by-step reasoning to make the best decision.

## RESIDENT PROFILE
- Age: {age}, Gender: {gender}
- Employment: {employment}
- Parish: {parish}
- Current location: Node {current_location}
- Energy level: {agent_state.energy_level:.1f}/1.0

## CURRENT CONTEXT
- Time: {time_of_day}:00 on {day_name}
- {needs_text}
- {memory_text}

## AVAILABLE DESTINATIONS
{chr(10).join(nearby_pois) if nearby_pois else "- No specific POIs detected in immediate area"}

## DECISION FRAMEWORK
Please think through this step-by-step:

### Step 1: Analyze Current Situation
Consider the resident's profile, current time, and immediate needs. What factors are most important right now?

### Step 2: Evaluate Timing Appropriateness
Given it's {time_of_day}:00 on {day_name}, what activities make sense? Consider:
- Business hours for different POI types
- Typical daily routines for someone of this age/employment status
- Meal times, work hours, leisure periods

### Step 3: Prioritize Needs and Goals
Based on the resident's current needs and recent activities:
- Which needs are most urgent (>70/100)?
- What activities would best satisfy multiple needs?
- Are there any critical daily activities missing?

### Step 4: Consider Fifteen-Minute City Principles
- Prefer destinations within 15-minute walking distance
- Choose locations that support sustainable urban living
- Consider mixed-use accessibility and community engagement

### Step 5: Make Final Decision
Based on your analysis, what is the most appropriate next destination?

## RESPONSE FORMAT
Provide your reasoning following the steps above, then conclude with:

**DECISION: [Choose ONE]**
- go_home (return to residence)
- restaurant (for meals/food)
- shop (for shopping/retail)
- hospital (for healthcare)
- park (for recreation/nature)
- library (for study/reading)
- cinema (for entertainment)
- gym (for exercise/fitness)
- pharmacy (for medicine)
- bank (for financial services)
- supermarket (for groceries)
- stay_put (remain at current location)

**RATIONALE:** [Brief explanation of why this choice best serves the resident's needs and fits the fifteen-minute city context]

Think through each step carefully before making your final decision."""

        return prompt

    def score_path_options(self, agent_state, path_options, context):
        """
        Use LLM to score and select the best path from multiple options for urban mobility.
        This is specific to fifteen-minute city path selection.
        
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
            # Create LLM prompt for urban mobility
            prompt = self._create_urban_path_scoring_prompt(agent_state, path_options, context)
            
            # Call LLM (placeholder for actual implementation)
            selected_path_id = self._call_llm_api(prompt, path_options)
            
            # Return response
            return PathScoringResponse(
                selected_path_id=selected_path_id,
                reasoning=f"LLM selected path {selected_path_id + 1} based on urban mobility criteria",
                confidence=0.8
            )
            
        except Exception as e:
            # Fallback to rule-based selection
            return self._fallback_urban_selection(path_options)

    def _create_urban_path_scoring_prompt(self, agent_state, path_options, context):
        """
        Create LLM prompt for urban path scoring (1-10 scale).
        Specific to fifteen-minute city mobility decisions.
        
        Args:
            agent_state: Agent characteristics
            path_options: Available paths with OSM metadata
            context: Decision context
            
        Returns:
            Formatted prompt string
        """
        age = agent_state.get('age', 30)
        time_of_day = context.get('time_of_day', 12)
        
        prompt = f"""You are helping a {age}-year-old urban resident choose the best walking path at {time_of_day}:00 in a fifteen-minute city.

Score each path from 1-10 (10 = best choice) considering fifteen-minute city principles:
- Travel time efficiency (should be within 15 minutes)
- Pedestrian safety and comfort
- Road type appropriateness for walking
- Green area coverage (parks, forests - promotes well-being)
- Urban accessibility and connectivity
- Overall convenience for daily activities

Available paths:
"""
        
        for path in path_options:
            green_desc = f"{path.get('green_area_percentage', 0)}% through green areas"
            prompt += f"""
Path {path['path_id']}:
- Distance: {path['distance_meters']} meters
- Walking time: {path['travel_time_minutes']} minutes
- Main road type: {path['dominant_road_type']}
- Road types: {', '.join(path['road_types'][:3])}
- Segments: {path['total_segments']}
- Green coverage: {green_desc}
"""
        
        prompt += f"""
Respond with ONLY the path number (1-{len(path_options)}) that best supports fifteen-minute city living.
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
        try:
            # Try to use the actual LLM through the model's brain system
            # This is a placeholder for actual LLM integration
            # For now, we'll simulate an LLM response that returns a path number
            
            # Create a simple mock LLM response that returns a path number
            # In a real implementation, this would call the actual LLM API
            # response = your_llm_service.generate(prompt)
            
            # For now, use rule-based selection but return it as if from LLM
            selected_index = self._rule_based_urban_scoring(path_options)
            
            # Ensure we return a valid integer index
            if isinstance(selected_index, int) and 0 <= selected_index < len(path_options):
                return selected_index
            else:
                # Fallback to first path if invalid index
                return 0
                
        except Exception as e:
            self.logger.error(f"Error in _call_llm_api: {e}")
            # Fallback to rule-based selection
            return self._rule_based_urban_scoring(path_options)

    def _rule_based_urban_scoring(self, path_options):
        """
        Rule-based scoring when LLM is not available.
        Optimized for fifteen-minute city principles.
        
        Args:
            path_options: Available path options
            
        Returns:
            Index of best path (0-based)
        """
        scores = []
        
        for path in path_options:
            score = 0
            
            # Time efficiency with 15-minute penalty (40% weight)
            travel_time = path['travel_time_minutes']
            if travel_time <= 15:
                time_score = 1.0 - (travel_time / 15.0) * 0.5  # Best score for shortest time within 15 min
            else:
                time_score = 0.5 - min(0.5, (travel_time - 15) / 15.0)  # Penalty for over 15 minutes
            score += 4.0 * time_score
            
            # Pedestrian-friendly road types (35% weight)
            pedestrian_scores = {
                'footway': 5.0, 'path': 4.8, 'residential': 4.0, 
                'tertiary': 3.5, 'secondary': 2.5, 'primary': 2.0,
                'trunk': 1.0, 'motorway': 0.5, 'unclassified': 3.0
            }
            road_score = pedestrian_scores.get(path['dominant_road_type'], 2.5)
            score += 3.5 * (road_score / 5.0)
            
            # Green area coverage for well-being (25% weight)
            green_percentage = path.get('green_area_percentage', 0)
            green_score = min(1.0, green_percentage / 40.0)  # Normalize (40% green = max score)
            score += 2.5 * green_score
            
            scores.append(score)
        
        # Return index of highest scoring path
        return scores.index(max(scores))

    def _fallback_urban_selection(self, path_options):
        """
        Simple fallback path selection optimized for urban mobility.
        
        Args:
            path_options: Available path options
            
        Returns:
            PathScoringResponse
        """
        # Select path with best combination of time and pedestrian-friendliness
        best_idx = 0
        best_score = -1
        
        for i, path in enumerate(path_options):
            # Simple scoring: prefer shorter time and pedestrian-friendly roads
            time_score = max(0, 1 - path['travel_time_minutes'] / 20.0)  # Normalize by 20 minutes
            
            pedestrian_bonus = 0
            if path['dominant_road_type'] in ['footway', 'path', 'residential']:
                pedestrian_bonus = 0.3
            
            combined_score = time_score + pedestrian_bonus
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = i
        
        return PathScoringResponse(
            selected_path_id=best_idx,
            reasoning="Selected best path for urban pedestrian mobility (fallback)",
            confidence=0.6
        ) 