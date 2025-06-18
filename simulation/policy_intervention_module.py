#!/usr/bin/env python3
"""
Policy Intervention Module for ABM Healthcare Policy Simulation

This module provides mechanisms for introducing, scheduling, and implementing diverse 
healthcare policy scenarios within the simulation. It combines rule-based efficiency 
for tangible changes with LLM-driven reasoning for complex behavioral responses.

Components:
1. Policy Definition Framework: Structured policy representation
2. Rule-Based Mechanisms: Direct attribute/environment modifications
3. LLM-Driven Mechanisms: Reasoning-based decision logic changes
4. Policy Scheduler: Timing and coordination of interventions
5. Impact Assessment: Monitoring and evaluation of policy effects
"""

import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

from agents.persona_memory_modules import PersonaType, PersonaMemoryManager
from simulation.llm_interaction_layer import LLMInteractionLayer


class PolicyMechanism(Enum):
    """Core mechanisms for policy implementation."""
    ATTRIBUTE_MODIFICATION = "attribute_modification"  # Direct state changes
    DECISION_LOGIC_CHANGE = "decision_logic_change"   # Behavioral reasoning changes
    ENVIRONMENT_SHIFT = "environment_shift"           # Environmental parameter changes
    INFORMATION_CHANGE = "information_change"         # Information/awareness campaigns


class PolicyScope(Enum):
    """Scope of policy application."""
    UNIVERSAL = "universal"          # All agents
    TARGETED = "targeted"           # Specific agent groups
    GEOGRAPHIC = "geographic"       # Location-based
    CONDITIONAL = "conditional"     # Based on agent characteristics


class PolicyStatus(Enum):
    """Status of policy implementation."""
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class PolicyTarget:
    """Defines who/what a policy targets."""
    scope: PolicyScope
    criteria: Dict[str, Any]  # Targeting criteria
    exclusions: List[str] = field(default_factory=list)  # Exclusion criteria


@dataclass
class PolicyEffect:
    """Defines the effect of a policy intervention."""
    mechanism: PolicyMechanism
    primary_approach: str  # "rule_based" or "llm_driven"
    parameters: Dict[str, Any]
    duration: Optional[int] = None  # Duration in time steps, None for permanent
    intensity: float = 1.0  # Effect intensity (0.0-1.0)


@dataclass
class PolicyIntervention:
    """Complete policy intervention definition."""
    policy_id: str
    name: str
    description: str
    category: str  # e.g., "healthcare_access", "preventive_care", "emergency_response"
    target: PolicyTarget
    effects: List[PolicyEffect]
    start_time: int  # Simulation time step
    end_time: Optional[int] = None  # None for permanent policies
    status: PolicyStatus = PolicyStatus.SCHEDULED
    metadata: Dict[str, Any] = field(default_factory=dict)


class RuleBasedPolicyEngine:
    """Handles rule-based policy implementations (direct modifications)."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def apply_attribute_modification(self, agents: List[Any], effect: PolicyEffect, policy: PolicyIntervention):
        """Apply direct attribute modifications to agents."""
        params = effect.parameters
        attribute = params.get('attribute')
        change_type = params.get('change_type', 'add')  # 'add', 'multiply', 'set'
        value = params.get('value', 0)
        
        affected_agents = 0
        
        for agent in agents:
            if self._agent_matches_target(agent, policy.target):
                try:
                    current_value = getattr(agent, attribute, 0)
                    
                    if change_type == 'add':
                        new_value = current_value + (value * effect.intensity)
                    elif change_type == 'multiply':
                        new_value = current_value * (1 + value * effect.intensity)
                    elif change_type == 'set':
                        new_value = value * effect.intensity
                    else:
                        new_value = current_value
                    
                    setattr(agent, attribute, new_value)
                    affected_agents += 1
                    
                    self.logger.debug(f"Modified {attribute} for agent {agent.unique_id}: {current_value} -> {new_value}")
                    
                except AttributeError:
                    self.logger.warning(f"Agent {agent.unique_id} does not have attribute {attribute}")
        
        self.logger.info(f"Applied attribute modification to {affected_agents} agents for policy {policy.name}")
    
    def apply_environment_shift(self, model: Any, effect: PolicyEffect, policy: PolicyIntervention):
        """Apply environmental changes to the simulation model."""
        params = effect.parameters
        env_variable = params.get('variable')
        change_type = params.get('change_type', 'set')
        value = params.get('value')
        
        try:
            if hasattr(model, env_variable):
                current_value = getattr(model, env_variable)
                
                if change_type == 'add':
                    new_value = current_value + (value * effect.intensity)
                elif change_type == 'multiply':
                    new_value = current_value * (1 + value * effect.intensity)
                elif change_type == 'set':
                    new_value = value
                else:
                    new_value = current_value
                
                setattr(model, env_variable, new_value)
                self.logger.info(f"Environmental shift: {env_variable} changed to {new_value} for policy {policy.name}")
            
            # Handle geographic/location-specific changes
            if 'location_changes' in params:
                location_changes = params['location_changes']
                for location, changes in location_changes.items():
                    # Apply location-specific environmental changes
                    self._apply_location_changes(model, location, changes, effect.intensity)
        
        except Exception as e:
            self.logger.error(f"Error applying environment shift for policy {policy.name}: {e}")
    
    def _apply_location_changes(self, model: Any, location: str, changes: Dict[str, Any], intensity: float):
        """Apply changes to specific locations in the model."""
        # This would interact with your spatial model
        # Example: modify accessibility, capacity, or availability of healthcare facilities
        if hasattr(model, 'healthcare_facilities'):
            facilities = getattr(model, 'healthcare_facilities', {})
            if location in facilities:
                facility = facilities[location]
                for attr, value in changes.items():
                    if hasattr(facility, attr):
                        current = getattr(facility, attr)
                        new_value = current + (value * intensity)
                        setattr(facility, attr, max(0, new_value))  # Ensure non-negative
    
    def _agent_matches_target(self, agent: Any, target: PolicyTarget) -> bool:
        """Check if an agent matches the policy target criteria."""
        if target.scope == PolicyScope.UNIVERSAL:
            return True
        
        criteria = target.criteria
        
        # Check persona type targeting
        if 'persona_types' in criteria:
            agent_persona = getattr(agent, 'persona_type', None)
            if agent_persona and agent_persona not in criteria['persona_types']:
                return False
        
        # Check demographic targeting
        if 'demographics' in criteria:
            for demo_key, demo_values in criteria['demographics'].items():
                agent_value = getattr(agent, demo_key, None)
                if agent_value not in demo_values:
                    return False
        
        # Check geographic targeting
        if target.scope == PolicyScope.GEOGRAPHIC and 'locations' in criteria:
            agent_location = getattr(agent, 'current_location', None) or getattr(agent, 'parish', None)
            if agent_location not in criteria['locations']:
                return False
        
        # Check exclusions
        for exclusion in target.exclusions:
            if hasattr(agent, exclusion):
                return False
        
        return True


class LLMDrivenPolicyEngine:
    """Handles LLM-driven policy implementations (reasoning-based changes)."""
    
    def __init__(self, llm_layer: LLMInteractionLayer, persona_manager: PersonaMemoryManager):
        self.llm_layer = llm_layer
        self.persona_manager = persona_manager
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def apply_decision_logic_change(self, agents: List[Any], effect: PolicyEffect, policy: PolicyIntervention):
        """Apply decision logic changes through LLM reasoning."""
        params = effect.parameters
        new_guidelines = params.get('guidelines', '')
        priority_changes = params.get('priority_changes', {})
        constraints = params.get('constraints', [])
        
        affected_agents = 0
        
        for agent in agents:
            if self._agent_matches_target(agent, policy.target):
                try:
                    # Create policy awareness experience
                    policy_experience = {
                        "type": "policy_change",
                        "policy_name": policy.name,
                        "description": policy.description,
                        "guidelines": new_guidelines,
                        "priority_changes": priority_changes,
                        "constraints": constraints,
                        "outcome": "informed",
                        "satisfaction": 0.5,  # Neutral initial reaction
                        "details": f"New policy: {policy.description}"
                    }
                    
                    # Update agent's memory with policy information
                    agent_id = str(agent.unique_id)
                    self.persona_manager.update_agent_experience(agent_id, policy_experience)
                    
                    # Store policy-specific decision modifications for future LLM calls
                    if not hasattr(agent, 'policy_contexts'):
                        agent.policy_contexts = {}
                    
                    agent.policy_contexts[policy.policy_id] = {
                        'guidelines': new_guidelines,
                        'priority_changes': priority_changes,
                        'constraints': constraints,
                        'active_since': policy.start_time
                    }
                    
                    affected_agents += 1
                    self.logger.debug(f"Applied decision logic change to agent {agent.unique_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error applying decision logic change to agent {agent.unique_id}: {e}")
        
        self.logger.info(f"Applied decision logic changes to {affected_agents} agents for policy {policy.name}")
    
    def apply_information_change(self, agents: List[Any], effect: PolicyEffect, policy: PolicyIntervention):
        """Apply information campaigns and awareness changes."""
        params = effect.parameters
        campaign_message = params.get('message', '')
        information_type = params.get('type', 'awareness')  # 'awareness', 'education', 'warning'
        credibility = params.get('credibility', 0.8)
        channels = params.get('channels', ['general'])
        
        affected_agents = 0
        
        for agent in agents:
            if self._agent_matches_target(agent, policy.target):
                try:
                    # Determine agent's receptivity based on persona
                    agent_id = str(agent.unique_id)
                    persona_type = getattr(agent, 'persona_type', PersonaType.YOUNG_PROFESSIONAL)
                    
                    # Get agent's current emotional state
                    context = self.persona_manager.get_agent_context_summary(agent_id)
                    
                    # Calculate information acceptance based on persona and state
                    acceptance_rate = self._calculate_information_acceptance(
                        persona_type, context, information_type, credibility, channels
                    )
                    
                    if np.random.random() < acceptance_rate:
                        # Agent accepts and processes the information
                        info_experience = {
                            "type": "information_campaign",
                            "policy_name": policy.name,
                            "message": campaign_message,
                            "information_type": information_type,
                            "channels": channels,
                            "outcome": "accepted",
                            "satisfaction": min(1.0, credibility + 0.1),
                            "details": f"Received information: {campaign_message}"
                        }
                        
                        # Update knowledge base access if relevant
                        if information_type == 'education':
                            self._update_agent_knowledge_access(agent, policy, campaign_message)
                    
                    else:
                        # Agent rejects or ignores the information
                        info_experience = {
                            "type": "information_campaign",
                            "policy_name": policy.name,
                            "message": campaign_message,
                            "outcome": "rejected",
                            "satisfaction": 0.3,
                            "details": f"Skeptical about: {campaign_message}"
                        }
                    
                    self.persona_manager.update_agent_experience(agent_id, info_experience)
                    affected_agents += 1
                    
                except Exception as e:
                    self.logger.error(f"Error applying information change to agent {agent.unique_id}: {e}")
        
        self.logger.info(f"Applied information campaign to {affected_agents} agents for policy {policy.name}")
    
    def _calculate_information_acceptance(self, persona_type: PersonaType, context: Dict[str, Any], 
                                        info_type: str, credibility: float, channels: List[str]) -> float:
        """Calculate the probability that an agent accepts information based on their persona."""
        base_acceptance = credibility
        
        # Adjust based on persona type
        if persona_type == PersonaType.ELDERLY_RESIDENT:
            # More trusting of official sources, less of digital channels
            if 'official' in channels or 'healthcare_professional' in channels:
                base_acceptance += 0.2
            if 'digital' in channels or 'social_media' in channels:
                base_acceptance -= 0.3
        
        elif persona_type == PersonaType.YOUNG_PROFESSIONAL:
            # More receptive to digital channels, data-driven information
            if 'digital' in channels or 'data_driven' in info_type:
                base_acceptance += 0.2
            if info_type == 'warning' and credibility < 0.7:
                base_acceptance -= 0.1  # Skeptical of low-credibility warnings
        
        elif persona_type == PersonaType.CHRONIC_PATIENT:
            # Highly receptive to health-related information
            if 'health' in info_type or 'medical' in channels:
                base_acceptance += 0.3
        
        elif persona_type == PersonaType.WORKING_PARENT:
            # Pragmatic, focused on family-relevant information
            if 'family' in info_type or 'practical' in info_type:
                base_acceptance += 0.2
        
        # Adjust based on current emotional state
        stress_level = context.get('stress_level', 0.5)
        trust_level = context.get('trust_in_healthcare', 0.7)
        
        # High stress can make agents more or less receptive depending on info type
        if info_type == 'warning' and stress_level > 0.7:
            base_acceptance += 0.1  # More receptive to warnings when stressed
        elif stress_level > 0.8:
            base_acceptance -= 0.1  # General information overload when very stressed
        
        # Trust in healthcare affects acceptance of medical information
        if 'health' in info_type or 'medical' in channels:
            base_acceptance += (trust_level - 0.5) * 0.3
        
        return max(0.0, min(1.0, base_acceptance))
    
    def _update_agent_knowledge_access(self, agent: Any, policy: PolicyIntervention, message: str):
        """Update agent's knowledge access based on educational campaigns."""
        if not hasattr(agent, 'policy_knowledge'):
            agent.policy_knowledge = {}
        
        agent.policy_knowledge[policy.policy_id] = {
            'message': message,
            'acquired_at': policy.start_time,
            'policy_name': policy.name
        }
    
    def _agent_matches_target(self, agent: Any, target: PolicyTarget) -> bool:
        """Check if an agent matches the policy target criteria (same as rule-based)."""
        if target.scope == PolicyScope.UNIVERSAL:
            return True
        
        criteria = target.criteria
        
        # Check persona type targeting
        if 'persona_types' in criteria:
            agent_persona = getattr(agent, 'persona_type', None)
            if agent_persona and agent_persona not in criteria['persona_types']:
                return False
        
        # Check demographic targeting
        if 'demographics' in criteria:
            for demo_key, demo_values in criteria['demographics'].items():
                agent_value = getattr(agent, demo_key, None)
                if agent_value not in demo_values:
                    return False
        
        # Check geographic targeting
        if target.scope == PolicyScope.GEOGRAPHIC and 'locations' in criteria:
            agent_location = getattr(agent, 'current_location', None) or getattr(agent, 'parish', None)
            if agent_location not in criteria['locations']:
                return False
        
        return True


class PolicyScheduler:
    """Manages timing and coordination of policy interventions."""
    
    def __init__(self):
        self.scheduled_policies: List[PolicyIntervention] = []
        self.active_policies: List[PolicyIntervention] = []
        self.completed_policies: List[PolicyIntervention] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def schedule_policy(self, policy: PolicyIntervention):
        """Schedule a policy for future implementation."""
        self.scheduled_policies.append(policy)
        self.scheduled_policies.sort(key=lambda p: p.start_time)
        self.logger.info(f"Scheduled policy '{policy.name}' for time step {policy.start_time}")
    
    def update(self, current_time: int) -> Tuple[List[PolicyIntervention], List[PolicyIntervention]]:
        """Update policy statuses and return policies to activate/deactivate."""
        to_activate = []
        to_deactivate = []
        
        # Check for policies to activate
        for policy in self.scheduled_policies[:]:
            if policy.start_time <= current_time:
                policy.status = PolicyStatus.ACTIVE
                self.active_policies.append(policy)
                self.scheduled_policies.remove(policy)
                to_activate.append(policy)
                self.logger.info(f"Activated policy '{policy.name}' at time step {current_time}")
        
        # Check for policies to deactivate
        for policy in self.active_policies[:]:
            if policy.end_time is not None and policy.end_time <= current_time:
                policy.status = PolicyStatus.COMPLETED
                self.completed_policies.append(policy)
                self.active_policies.remove(policy)
                to_deactivate.append(policy)
                self.logger.info(f"Completed policy '{policy.name}' at time step {current_time}")
        
        return to_activate, to_deactivate
    
    def get_active_policies(self) -> List[PolicyIntervention]:
        """Get all currently active policies."""
        return self.active_policies.copy()
    
    def cancel_policy(self, policy_id: str):
        """Cancel a scheduled or active policy."""
        # Check scheduled policies
        for policy in self.scheduled_policies[:]:
            if policy.policy_id == policy_id:
                policy.status = PolicyStatus.CANCELLED
                self.scheduled_policies.remove(policy)
                self.completed_policies.append(policy)
                self.logger.info(f"Cancelled scheduled policy '{policy.name}'")
                return
        
        # Check active policies
        for policy in self.active_policies[:]:
            if policy.policy_id == policy_id:
                policy.status = PolicyStatus.CANCELLED
                self.active_policies.remove(policy)
                self.completed_policies.append(policy)
                self.logger.info(f"Cancelled active policy '{policy.name}'")
                return


class PolicyImpactAssessment:
    """Monitors and evaluates the effects of policy interventions."""
    
    def __init__(self, persona_manager: PersonaMemoryManager):
        self.persona_manager = persona_manager
        self.policy_metrics: Dict[str, Dict[str, Any]] = {}
        self.baseline_metrics: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def establish_baseline(self, agents: List[Any], metrics: List[str]):
        """Establish baseline metrics before policy implementation."""
        self.baseline_metrics = self._collect_metrics(agents, metrics)
        self.logger.info(f"Established baseline metrics: {list(self.baseline_metrics.keys())}")
    
    def assess_policy_impact(self, policy: PolicyIntervention, agents: List[Any], 
                           current_time: int, metrics: List[str]) -> Dict[str, Any]:
        """Assess the impact of a specific policy."""
        current_metrics = self._collect_metrics(agents, metrics)
        
        if policy.policy_id not in self.policy_metrics:
            self.policy_metrics[policy.policy_id] = {
                'policy_name': policy.name,
                'start_time': policy.start_time,
                'measurements': []
            }
        
        # Calculate changes from baseline
        impact_assessment = {
            'time_step': current_time,
            'metrics': current_metrics,
            'changes_from_baseline': {},
            'affected_agents': self._count_affected_agents(agents, policy)
        }
        
        for metric, current_value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    change = current_value - baseline_value
                    percent_change = (change / baseline_value * 100) if baseline_value != 0 else 0
                    impact_assessment['changes_from_baseline'][metric] = {
                        'absolute_change': change,
                        'percent_change': percent_change
                    }
        
        self.policy_metrics[policy.policy_id]['measurements'].append(impact_assessment)
        return impact_assessment
    
    def _collect_metrics(self, agents: List[Any], metrics: List[str]) -> Dict[str, Any]:
        """Collect specified metrics from agents."""
        collected_metrics = {}
        
        for metric in metrics:
            if metric == 'average_satisfaction':
                satisfactions = []
                for agent in agents:
                    agent_id = str(agent.unique_id)
                    context = self.persona_manager.get_agent_context_summary(agent_id)
                    if 'satisfaction_with_services' in context:
                        satisfactions.append(context['satisfaction_with_services'])
                collected_metrics[metric] = np.mean(satisfactions) if satisfactions else 0.0
            
            elif metric == 'average_stress':
                stress_levels = []
                for agent in agents:
                    agent_id = str(agent.unique_id)
                    context = self.persona_manager.get_agent_context_summary(agent_id)
                    if 'stress_level' in context:
                        stress_levels.append(context['stress_level'])
                collected_metrics[metric] = np.mean(stress_levels) if stress_levels else 0.0
            
            elif metric == 'trust_in_healthcare':
                trust_levels = []
                for agent in agents:
                    agent_id = str(agent.unique_id)
                    context = self.persona_manager.get_agent_context_summary(agent_id)
                    if 'trust_in_healthcare' in context:
                        trust_levels.append(context['trust_in_healthcare'])
                collected_metrics[metric] = np.mean(trust_levels) if trust_levels else 0.0
            
            elif metric == 'healthcare_utilization':
                # Count agents who have recent healthcare experiences
                utilization_count = 0
                for agent in agents:
                    agent_id = str(agent.unique_id)
                    state = self.persona_manager.emotional_tracker.get_agent_state(agent_id)
                    if state and state.recent_experiences:
                        for exp in state.recent_experiences:
                            if exp.get('type') == 'healthcare_visit':
                                utilization_count += 1
                                break
                collected_metrics[metric] = utilization_count
        
        return collected_metrics
    
    def _count_affected_agents(self, agents: List[Any], policy: PolicyIntervention) -> int:
        """Count how many agents are affected by a policy."""
        count = 0
        for agent in agents:
            if self._agent_matches_target(agent, policy.target):
                count += 1
        return count
    
    def _agent_matches_target(self, agent: Any, target: PolicyTarget) -> bool:
        """Check if an agent matches the policy target criteria."""
        if target.scope == PolicyScope.UNIVERSAL:
            return True
        
        criteria = target.criteria
        
        if 'persona_types' in criteria:
            agent_persona = getattr(agent, 'persona_type', None)
            if agent_persona and agent_persona not in criteria['persona_types']:
                return False
        
        return True
    
    def generate_policy_report(self, policy_id: str) -> Dict[str, Any]:
        """Generate a comprehensive report for a specific policy."""
        if policy_id not in self.policy_metrics:
            return {"error": "Policy not found"}
        
        policy_data = self.policy_metrics[policy_id]
        measurements = policy_data['measurements']
        
        if not measurements:
            return {"error": "No measurements available"}
        
        # Calculate summary statistics
        latest_measurement = measurements[-1]
        first_measurement = measurements[0]
        
        report = {
            'policy_name': policy_data['policy_name'],
            'policy_id': policy_id,
            'start_time': policy_data['start_time'],
            'total_measurements': len(measurements),
            'affected_agents': latest_measurement['affected_agents'],
            'latest_metrics': latest_measurement['metrics'],
            'impact_summary': latest_measurement['changes_from_baseline'],
            'trend_analysis': self._analyze_trends(measurements)
        }
        
        return report
    
    def _analyze_trends(self, measurements: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze trends in policy impact over time."""
        if len(measurements) < 2:
            return {"status": "Insufficient data for trend analysis"}
        
        trends = {}
        first = measurements[0]
        latest = measurements[-1]
        
        for metric in first.get('changes_from_baseline', {}):
            first_change = first['changes_from_baseline'][metric]['percent_change']
            latest_change = latest['changes_from_baseline'][metric]['percent_change']
            
            if latest_change > first_change + 5:
                trends[metric] = "improving"
            elif latest_change < first_change - 5:
                trends[metric] = "declining"
            else:
                trends[metric] = "stable"
        
        return trends


class PolicyInterventionModule:
    """Main interface for the Policy Intervention Module."""
    
    def __init__(self, llm_layer: LLMInteractionLayer, persona_manager: PersonaMemoryManager):
        self.rule_based_engine = RuleBasedPolicyEngine()
        self.llm_driven_engine = LLMDrivenPolicyEngine(llm_layer, persona_manager)
        self.scheduler = PolicyScheduler()
        self.impact_assessor = PolicyImpactAssessment(persona_manager)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Policy Intervention Module initialized")
    
    def create_healthcare_policy(self, policy_config: Dict[str, Any]) -> PolicyIntervention:
        """Create a healthcare policy from configuration."""
        # Create target
        target = PolicyTarget(
            scope=PolicyScope(policy_config['target']['scope']),
            criteria=policy_config['target']['criteria'],
            exclusions=policy_config['target'].get('exclusions', [])
        )
        
        # Create effects
        effects = []
        for effect_config in policy_config['effects']:
            effect = PolicyEffect(
                mechanism=PolicyMechanism(effect_config['mechanism']),
                primary_approach=effect_config['primary_approach'],
                parameters=effect_config['parameters'],
                duration=effect_config.get('duration'),
                intensity=effect_config.get('intensity', 1.0)
            )
            effects.append(effect)
        
        # Create policy
        policy = PolicyIntervention(
            policy_id=policy_config['policy_id'],
            name=policy_config['name'],
            description=policy_config['description'],
            category=policy_config['category'],
            target=target,
            effects=effects,
            start_time=policy_config['start_time'],
            end_time=policy_config.get('end_time'),
            metadata=policy_config.get('metadata', {})
        )
        
        return policy
    
    def implement_policy(self, policy: PolicyIntervention, agents: List[Any], model: Any = None):
        """Implement a policy using appropriate mechanisms."""
        self.logger.info(f"Implementing policy: {policy.name}")
        
        for effect in policy.effects:
            try:
                if effect.mechanism == PolicyMechanism.ATTRIBUTE_MODIFICATION:
                    if effect.primary_approach == "rule_based":
                        self.rule_based_engine.apply_attribute_modification(agents, effect, policy)
                    # Could add LLM-driven attribute modification for complex cases
                
                elif effect.mechanism == PolicyMechanism.DECISION_LOGIC_CHANGE:
                    if effect.primary_approach == "llm_driven":
                        self.llm_driven_engine.apply_decision_logic_change(agents, effect, policy)
                    # Could add rule-based decision logic for simple cases
                
                elif effect.mechanism == PolicyMechanism.ENVIRONMENT_SHIFT:
                    if effect.primary_approach == "rule_based" and model:
                        self.rule_based_engine.apply_environment_shift(model, effect, policy)
                    # Could add LLM-driven environment interpretation
                
                elif effect.mechanism == PolicyMechanism.INFORMATION_CHANGE:
                    if effect.primary_approach == "llm_driven":
                        self.llm_driven_engine.apply_information_change(agents, effect, policy)
                    # Could add rule-based information broadcasting
                
            except Exception as e:
                self.logger.error(f"Error implementing effect {effect.mechanism.value} for policy {policy.name}: {e}")
    
    def schedule_policy(self, policy: PolicyIntervention):
        """Schedule a policy for future implementation."""
        self.scheduler.schedule_policy(policy)
    
    def update_policies(self, current_time: int, agents: List[Any], model: Any = None):
        """Update policy statuses and implement new policies."""
        to_activate, to_deactivate = self.scheduler.update(current_time)
        
        # Implement newly activated policies
        for policy in to_activate:
            self.implement_policy(policy, agents, model)
        
        # Handle policy deactivation if needed
        for policy in to_deactivate:
            self.logger.info(f"Policy '{policy.name}' has ended")
    
    def assess_policy_impacts(self, agents: List[Any], current_time: int, 
                            metrics: List[str] = None) -> Dict[str, Any]:
        """Assess impacts of all active policies."""
        if metrics is None:
            metrics = ['average_satisfaction', 'average_stress', 'trust_in_healthcare', 'healthcare_utilization']
        
        assessments = {}
        active_policies = self.scheduler.get_active_policies()
        
        for policy in active_policies:
            assessment = self.impact_assessor.assess_policy_impact(policy, agents, current_time, metrics)
            assessments[policy.policy_id] = assessment
        
        return assessments
    
    def generate_policy_report(self, policy_id: str) -> Dict[str, Any]:
        """Generate a comprehensive report for a specific policy."""
        return self.impact_assessor.generate_policy_report(policy_id)
    
    def establish_baseline_metrics(self, agents: List[Any], metrics: List[str] = None):
        """Establish baseline metrics before implementing policies."""
        if metrics is None:
            metrics = ['average_satisfaction', 'average_stress', 'trust_in_healthcare', 'healthcare_utilization']
        
        self.impact_assessor.establish_baseline(agents, metrics) 