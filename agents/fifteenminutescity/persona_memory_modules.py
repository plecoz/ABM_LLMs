"""
Simple Persona System for Agent-Based Models

This module provides basic demographic-based personas for explainability.
Focus on simple, interpretable characteristics rather than complex behavioral modeling.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging


@dataclass
class SimplePersona:
    """Simple persona based on basic demographics only."""
    name: str
    age: int
    household_type: str  # 'single', 'family', 'elderly'
    economic_status: str  # 'low', 'middle', 'high'
    description: str


class SimplePersonaManager:
    """Creates simple personas based on basic demographics."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Simple Persona Manager initialized")
    
    def create_persona(self, demographics: Dict[str, Any]) -> SimplePersona:
        """Create a simple persona from basic demographics."""
        
        age = demographics.get('age', 30)
        economic_status = demographics.get('economic_status', 'employed')
        household_type = demographics.get('household_type', 'single')
        parish = demographics.get('parish', 'unknown')
        
        # Simplify economic status to three categories
        if economic_status == 'Unemployed':
            econ_simple = 'low'
        elif economic_status in ['Employed', 'employed']:
            econ_simple = 'middle'
        else:
            econ_simple = 'middle'
        
        # Simplify household type
        if age >= 65:
            household_simple = 'elderly'
        elif 'family' in household_type.lower() or 'parent' in household_type.lower():
            household_simple = 'family'
        else:
            household_simple = 'single'
        
        # Create simple demographic placeholder for LLM interpretation
        name = f"Resident {age}y"
        description = f"Age: {age}, Household: {household_simple}, Income: {econ_simple}, Location: {parish}"
        
        return SimplePersona(
            name=name,
            age=age,
            household_type=household_simple,
            economic_status=econ_simple,
            description=description
        )


class PersonaMemoryManager:
    """Simplified main interface for persona management."""
    
    def __init__(self):
        self.persona_manager = SimplePersonaManager()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Simplified Persona Memory Manager initialized")
    
    def create_agent_persona(self, agent_id: str, demographics: Dict[str, Any]) -> Tuple[SimplePersona]:
        """Create a simple persona profile for an agent."""
        
        # Generate simple persona
        persona = self.persona_manager.create_persona(demographics)
        
        
        self.logger.debug(f"Created simple persona for agent {agent_id}: {persona.name}")
        return persona
    
    def update_agent_experience(self, agent_id: str, experience: Dict[str, Any]):
        """
        Simple placeholder for updating agent experience.
        In a full implementation, this would update the agent's emotional state and memories.
        
        Args:
            agent_id: The agent's unique identifier
            experience: Dictionary containing experience details
        """
        # Simple logging for now - no actual state updates needed for simplified system
        self.logger.debug(f"Agent {agent_id} had experience: {experience.get('type', 'unknown')} with outcome: {experience.get('outcome', 'unknown')}")
        # In the simplified system, we don't need to do anything else
        pass 