#!/usr/bin/env python3
"""
Data Management & Logging Toolkit for ABM Healthcare Policy Simulation

This module provides comprehensive data capture and logging capabilities for ABM simulations,
recording agent states, decisions, interactions, LLM invocations, spatial dynamics, and
environmental changes to enable debugging, auditing, and scientific analysis.

Components:
1. Agent State Logger: Captures dynamic agent states and decisions
2. LLM Interaction Logger: Documents LLM calls, prompts, responses, and metadata
3. Spatial Dynamics Logger: Tracks environmental maps and agent locations
4. Policy Effect Logger: Records policy implementations and impacts
5. Performance Monitor: Tracks simulation performance metrics
6. Data Export Manager: Handles data serialization and export formats
"""

import json
import logging
import sqlite3
import pickle
import csv
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import os
from pathlib import Path
import gzip
import threading
from queue import Queue
import time

from agents.persona_memory_modules import PersonaType, EmotionalState, MotivationType
from simulation.policy_intervention_module import PolicyIntervention


class LogLevel(Enum):
    """Logging levels for different types of data."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataFormat(Enum):
    """Supported data export formats."""
    JSON = "json"
    CSV = "csv"
    PICKLE = "pickle"
    HDF5 = "hdf5"
    SQLITE = "sqlite"
    PARQUET = "parquet"


@dataclass
class AgentStateRecord:
    """Record of agent state at a specific time step."""
    simulation_id: str
    time_step: int
    agent_id: str
    persona_type: str
    location: Tuple[float, float]
    parish: str
    
    # Core attributes
    wealth: float
    health_status: float
    age: int
    
    # Emotional and motivational state
    dominant_emotion: str
    primary_motivation: str
    stress_level: float
    confidence_level: float
    trust_in_healthcare: float
    satisfaction_with_services: float
    
    # Decision context
    current_activity: str
    recent_decisions: List[Dict[str, Any]]
    active_policies: List[str]
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMInteractionRecord:
    """Record of LLM interaction for debugging and analysis."""
    simulation_id: str
    time_step: int
    agent_id: str
    interaction_id: str
    
    # LLM call details
    model_name: str
    prompt_template: str
    full_prompt: str
    response: str
    
    # Context information
    agent_state: Dict[str, Any]
    decision_context: Dict[str, Any]
    policy_contexts: List[Dict[str, Any]]
    
    # Performance metrics
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimate: float
    
    # Quality metrics
    response_quality_score: Optional[float] = None
    coherence_score: Optional[float] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class SpatialDynamicsRecord:
    """Record of spatial state and agent movements."""
    simulation_id: str
    time_step: int
    
    # Agent locations
    agent_locations: Dict[str, Tuple[float, float]]
    agent_parishes: Dict[str, str]
    
    # Environmental state
    healthcare_facility_status: Dict[str, Dict[str, Any]]
    poi_accessibility: Dict[str, float]
    traffic_conditions: Dict[str, float]
    
    # Spatial interactions
    agent_interactions: List[Dict[str, Any]]
    facility_utilization: Dict[str, int]
    
    # Environmental changes
    environmental_changes: List[Dict[str, Any]]
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PolicyEffectRecord:
    """Record of policy implementation and effects."""
    simulation_id: str
    time_step: int
    policy_id: str
    policy_name: str
    
    # Implementation details
    mechanism: str
    primary_approach: str
    affected_agents: List[str]
    effect_parameters: Dict[str, Any]
    
    # Impact metrics
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    changes_from_baseline: Dict[str, Dict[str, float]]
    
    # Agent responses
    agent_acceptance_rates: Dict[str, float]
    behavioral_changes: Dict[str, List[Dict[str, Any]]]
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceRecord:
    """Record of simulation performance metrics."""
    simulation_id: str
    time_step: int
    
    # Timing metrics
    step_duration_ms: float
    agent_processing_time_ms: float
    llm_processing_time_ms: float
    policy_processing_time_ms: float
    
    # Resource usage
    memory_usage_mb: float
    cpu_usage_percent: float
    
    # LLM usage
    total_llm_calls: int
    total_tokens_used: int
    estimated_cost: float
    
    # Agent metrics
    active_agents: int
    decisions_made: int
    interactions_count: int
    
    timestamp: datetime = field(default_factory=datetime.now)


class AgentStateLogger:
    """Logs comprehensive agent state information."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # In-memory buffer for performance
        self.state_buffer: List[AgentStateRecord] = []
        self.buffer_size = 1000
        
    def log_agent_state(self, simulation_id: str, time_step: int, agent: Any, 
                       persona_manager: Any, additional_context: Dict[str, Any] = None):
        """Log the current state of an agent."""
        try:
            agent_id = str(agent.unique_id)
            
            # Get agent context from persona manager
            context = persona_manager.get_agent_context_summary(agent_id)
            
            # Extract location information
            location = getattr(agent, 'pos', (0.0, 0.0))
            if hasattr(agent, 'current_location'):
                parish = agent.current_location
            else:
                parish = getattr(agent, 'parish', 'Unknown')
            
            # Get recent decisions
            recent_decisions = []
            if hasattr(agent, 'decision_history'):
                recent_decisions = agent.decision_history[-5:]  # Last 5 decisions
            
            # Get active policies
            active_policies = []
            if hasattr(agent, 'policy_contexts'):
                active_policies = list(agent.policy_contexts.keys())
            
            # Create state record
            state_record = AgentStateRecord(
                simulation_id=simulation_id,
                time_step=time_step,
                agent_id=agent_id,
                persona_type=getattr(agent, 'persona_type', PersonaType.YOUNG_PROFESSIONAL).value,
                location=location,
                parish=parish,
                wealth=getattr(agent, 'wealth', 0.0),
                health_status=getattr(agent, 'health_status', 1.0),
                age=getattr(agent, 'age', 30),
                dominant_emotion=context.get('dominant_emotion', 'calm'),
                primary_motivation=context.get('primary_motivation', 'health_security'),
                stress_level=context.get('stress_level', 0.5),
                confidence_level=context.get('confidence_level', 0.5),
                trust_in_healthcare=context.get('trust_in_healthcare', 0.7),
                satisfaction_with_services=context.get('satisfaction_with_services', 0.5),
                current_activity=getattr(agent, 'current_activity', 'idle'),
                recent_decisions=recent_decisions,
                active_policies=active_policies
            )
            
            # Add to buffer
            self.state_buffer.append(state_record)
            
            # Flush buffer if full
            if len(self.state_buffer) >= self.buffer_size:
                self._flush_buffer()
                
        except Exception as e:
            self.logger.error(f"Error logging agent state for {agent.unique_id}: {e}")
    
    def _flush_buffer(self):
        """Flush the state buffer to disk."""
        if not self.state_buffer:
            return
        
        try:
            # Group by simulation_id and time_step for efficient storage
            grouped_data = {}
            for record in self.state_buffer:
                key = f"{record.simulation_id}_{record.time_step}"
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(asdict(record))
            
            # Write to files
            for key, records in grouped_data.items():
                filename = self.output_dir / f"agent_states_{key}.json.gz"
                with gzip.open(filename, 'wt') as f:
                    json.dump(records, f, default=str, indent=2)
            
            self.state_buffer.clear()
            self.logger.debug(f"Flushed {len(self.state_buffer)} agent state records")
            
        except Exception as e:
            self.logger.error(f"Error flushing agent state buffer: {e}")
    
    def finalize(self):
        """Finalize logging and flush remaining buffer."""
        self._flush_buffer()


class LLMInteractionLogger:
    """Logs LLM interactions for debugging and analysis."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # SQLite database for structured LLM interaction data
        self.db_path = self.output_dir / "llm_interactions.db"
        self._init_database()
        
        # In-memory buffer
        self.interaction_buffer: List[LLMInteractionRecord] = []
        self.buffer_size = 100
        
    def _init_database(self):
        """Initialize SQLite database for LLM interactions."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id TEXT,
                    time_step INTEGER,
                    agent_id TEXT,
                    interaction_id TEXT,
                    model_name TEXT,
                    prompt_template TEXT,
                    full_prompt TEXT,
                    response TEXT,
                    agent_state TEXT,
                    decision_context TEXT,
                    policy_contexts TEXT,
                    latency_ms REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_estimate REAL,
                    response_quality_score REAL,
                    coherence_score REAL,
                    timestamp TEXT,
                    success BOOLEAN,
                    error_message TEXT
                )
            ''')
            
            # Create indexes for efficient querying
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_simulation_time ON llm_interactions(simulation_id, time_step)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent ON llm_interactions(agent_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model ON llm_interactions(model_name)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM interaction database: {e}")
    
    def log_llm_interaction(self, simulation_id: str, time_step: int, agent_id: str,
                           interaction_data: Dict[str, Any]):
        """Log an LLM interaction."""
        try:
            record = LLMInteractionRecord(
                simulation_id=simulation_id,
                time_step=time_step,
                agent_id=agent_id,
                interaction_id=interaction_data.get('interaction_id', f"{agent_id}_{time_step}"),
                model_name=interaction_data.get('model_name', 'unknown'),
                prompt_template=interaction_data.get('prompt_template', ''),
                full_prompt=interaction_data.get('full_prompt', ''),
                response=interaction_data.get('response', ''),
                agent_state=interaction_data.get('agent_state', {}),
                decision_context=interaction_data.get('decision_context', {}),
                policy_contexts=interaction_data.get('policy_contexts', []),
                latency_ms=interaction_data.get('latency_ms', 0.0),
                input_tokens=interaction_data.get('input_tokens', 0),
                output_tokens=interaction_data.get('output_tokens', 0),
                total_tokens=interaction_data.get('total_tokens', 0),
                cost_estimate=interaction_data.get('cost_estimate', 0.0),
                response_quality_score=interaction_data.get('response_quality_score'),
                coherence_score=interaction_data.get('coherence_score'),
                success=interaction_data.get('success', True),
                error_message=interaction_data.get('error_message')
            )
            
            self.interaction_buffer.append(record)
            
            # Flush buffer if full
            if len(self.interaction_buffer) >= self.buffer_size:
                self._flush_buffer()
                
        except Exception as e:
            self.logger.error(f"Error logging LLM interaction: {e}")
    
    def _flush_buffer(self):
        """Flush the interaction buffer to database."""
        if not self.interaction_buffer:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for record in self.interaction_buffer:
                cursor.execute('''
                    INSERT INTO llm_interactions (
                        simulation_id, time_step, agent_id, interaction_id, model_name,
                        prompt_template, full_prompt, response, agent_state, decision_context,
                        policy_contexts, latency_ms, input_tokens, output_tokens, total_tokens,
                        cost_estimate, response_quality_score, coherence_score, timestamp,
                        success, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.simulation_id, record.time_step, record.agent_id, record.interaction_id,
                    record.model_name, record.prompt_template, record.full_prompt, record.response,
                    json.dumps(record.agent_state), json.dumps(record.decision_context),
                    json.dumps(record.policy_contexts), record.latency_ms, record.input_tokens,
                    record.output_tokens, record.total_tokens, record.cost_estimate,
                    record.response_quality_score, record.coherence_score, record.timestamp.isoformat(),
                    record.success, record.error_message
                ))
            
            conn.commit()
            conn.close()
            
            self.interaction_buffer.clear()
            self.logger.debug(f"Flushed {len(self.interaction_buffer)} LLM interaction records")
            
        except Exception as e:
            self.logger.error(f"Error flushing LLM interaction buffer: {e}")
    
    def finalize(self):
        """Finalize logging and flush remaining buffer."""
        self._flush_buffer()


class SpatialDynamicsLogger:
    """Logs spatial dynamics and environmental changes."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # HDF5 file for efficient spatial data storage
        self.hdf5_path = self.output_dir / "spatial_dynamics.h5"
        
    def log_spatial_state(self, simulation_id: str, time_step: int, model: Any, agents: List[Any]):
        """Log the spatial state of the simulation."""
        try:
            # Collect agent locations
            agent_locations = {}
            agent_parishes = {}
            
            for agent in agents:
                agent_id = str(agent.unique_id)
                agent_locations[agent_id] = getattr(agent, 'pos', (0.0, 0.0))
                agent_parishes[agent_id] = getattr(agent, 'parish', 'Unknown')
            
            # Collect facility status
            healthcare_facility_status = {}
            if hasattr(model, 'healthcare_facilities'):
                for facility_id, facility in model.healthcare_facilities.items():
                    healthcare_facility_status[facility_id] = {
                        'capacity': getattr(facility, 'capacity', 0),
                        'current_utilization': getattr(facility, 'current_utilization', 0),
                        'availability': getattr(facility, 'availability', 1.0),
                        'queue_length': getattr(facility, 'queue_length', 0)
                    }
            
            # Create spatial record
            spatial_record = SpatialDynamicsRecord(
                simulation_id=simulation_id,
                time_step=time_step,
                agent_locations=agent_locations,
                agent_parishes=agent_parishes,
                healthcare_facility_status=healthcare_facility_status,
                poi_accessibility={},  # Could be populated from model
                traffic_conditions={},  # Could be populated from model
                agent_interactions=[],  # Could track agent-agent interactions
                facility_utilization={},  # Could track facility usage
                environmental_changes=[]  # Could track environmental changes
            )
            
            # Store in HDF5 format for efficient access
            self._store_spatial_data(spatial_record)
            
        except Exception as e:
            self.logger.error(f"Error logging spatial state: {e}")
    
    def _store_spatial_data(self, record: SpatialDynamicsRecord):
        """Store spatial data in HDF5 format."""
        try:
            with h5py.File(self.hdf5_path, 'a') as f:
                # Create group for this time step
                group_name = f"{record.simulation_id}/time_{record.time_step}"
                group = f.create_group(group_name)
                
                # Store agent locations as arrays
                agent_ids = list(record.agent_locations.keys())
                locations = np.array([record.agent_locations[aid] for aid in agent_ids])
                
                group.create_dataset('agent_ids', data=[aid.encode() for aid in agent_ids])
                group.create_dataset('agent_locations', data=locations)
                
                # Store facility status
                if record.healthcare_facility_status:
                    facility_group = group.create_group('healthcare_facilities')
                    for facility_id, status in record.healthcare_facility_status.items():
                        fac_group = facility_group.create_group(facility_id)
                        for key, value in status.items():
                            fac_group.create_dataset(key, data=value)
                
                # Store metadata
                group.attrs['timestamp'] = record.timestamp.isoformat()
                
        except Exception as e:
            self.logger.error(f"Error storing spatial data: {e}")


class PolicyEffectLogger:
    """Logs policy implementations and their effects."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.policy_effects: List[PolicyEffectRecord] = []
        
    def log_policy_effect(self, simulation_id: str, time_step: int, policy: PolicyIntervention,
                         impact_data: Dict[str, Any], affected_agents: List[str]):
        """Log the effect of a policy implementation."""
        try:
            record = PolicyEffectRecord(
                simulation_id=simulation_id,
                time_step=time_step,
                policy_id=policy.policy_id,
                policy_name=policy.name,
                mechanism=policy.effects[0].mechanism.value if policy.effects else "unknown",
                primary_approach=policy.effects[0].primary_approach if policy.effects else "unknown",
                affected_agents=affected_agents,
                effect_parameters={effect.mechanism.value: effect.parameters for effect in policy.effects},
                baseline_metrics=impact_data.get('baseline_metrics', {}),
                current_metrics=impact_data.get('current_metrics', {}),
                changes_from_baseline=impact_data.get('changes_from_baseline', {}),
                agent_acceptance_rates=impact_data.get('agent_acceptance_rates', {}),
                behavioral_changes=impact_data.get('behavioral_changes', {})
            )
            
            self.policy_effects.append(record)
            
        except Exception as e:
            self.logger.error(f"Error logging policy effect: {e}")
    
    def export_policy_data(self, format: DataFormat = DataFormat.JSON):
        """Export policy effect data in specified format."""
        try:
            if format == DataFormat.JSON:
                filename = self.output_dir / "policy_effects.json"
                with open(filename, 'w') as f:
                    json.dump([asdict(record) for record in self.policy_effects], 
                             f, default=str, indent=2)
            
            elif format == DataFormat.CSV:
                filename = self.output_dir / "policy_effects.csv"
                df = pd.DataFrame([asdict(record) for record in self.policy_effects])
                df.to_csv(filename, index=False)
            
            self.logger.info(f"Exported policy effect data to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting policy data: {e}")


class PerformanceMonitor:
    """Monitors and logs simulation performance metrics."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.performance_records: List[PerformanceRecord] = []
        self.step_start_time = None
        
    def start_step_timing(self):
        """Start timing a simulation step."""
        self.step_start_time = time.time()
    
    def log_performance(self, simulation_id: str, time_step: int, 
                       performance_data: Dict[str, Any]):
        """Log performance metrics for a simulation step."""
        try:
            step_duration = (time.time() - self.step_start_time) * 1000 if self.step_start_time else 0
            
            record = PerformanceRecord(
                simulation_id=simulation_id,
                time_step=time_step,
                step_duration_ms=step_duration,
                agent_processing_time_ms=performance_data.get('agent_processing_time_ms', 0),
                llm_processing_time_ms=performance_data.get('llm_processing_time_ms', 0),
                policy_processing_time_ms=performance_data.get('policy_processing_time_ms', 0),
                memory_usage_mb=performance_data.get('memory_usage_mb', 0),
                cpu_usage_percent=performance_data.get('cpu_usage_percent', 0),
                total_llm_calls=performance_data.get('total_llm_calls', 0),
                total_tokens_used=performance_data.get('total_tokens_used', 0),
                estimated_cost=performance_data.get('estimated_cost', 0),
                active_agents=performance_data.get('active_agents', 0),
                decisions_made=performance_data.get('decisions_made', 0),
                interactions_count=performance_data.get('interactions_count', 0)
            )
            
            self.performance_records.append(record)
            
        except Exception as e:
            self.logger.error(f"Error logging performance metrics: {e}")
    
    def export_performance_data(self):
        """Export performance data for analysis."""
        try:
            filename = self.output_dir / "performance_metrics.csv"
            df = pd.DataFrame([asdict(record) for record in self.performance_records])
            df.to_csv(filename, index=False)
            
            self.logger.info(f"Exported performance data to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting performance data: {e}")


class DataExportManager:
    """Manages data export in various formats for analysis."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def export_simulation_data(self, simulation_id: str, data_sources: Dict[str, Any],
                              formats: List[DataFormat] = None):
        """Export complete simulation data in specified formats."""
        if formats is None:
            formats = [DataFormat.JSON, DataFormat.CSV]
        
        try:
            for format in formats:
                if format == DataFormat.JSON:
                    self._export_json(simulation_id, data_sources)
                elif format == DataFormat.CSV:
                    self._export_csv(simulation_id, data_sources)
                elif format == DataFormat.PICKLE:
                    self._export_pickle(simulation_id, data_sources)
                elif format == DataFormat.HDF5:
                    self._export_hdf5(simulation_id, data_sources)
                
        except Exception as e:
            self.logger.error(f"Error exporting simulation data: {e}")
    
    def _export_json(self, simulation_id: str, data_sources: Dict[str, Any]):
        """Export data in JSON format."""
        filename = self.output_dir / f"{simulation_id}_complete_data.json.gz"
        
        with gzip.open(filename, 'wt') as f:
            json.dump(data_sources, f, default=str, indent=2)
        
        self.logger.info(f"Exported JSON data to {filename}")
    
    def _export_csv(self, simulation_id: str, data_sources: Dict[str, Any]):
        """Export data in CSV format."""
        for data_type, data in data_sources.items():
            if isinstance(data, list) and data:
                filename = self.output_dir / f"{simulation_id}_{data_type}.csv"
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                self.logger.info(f"Exported {data_type} CSV to {filename}")
    
    def _export_pickle(self, simulation_id: str, data_sources: Dict[str, Any]):
        """Export data in pickle format."""
        filename = self.output_dir / f"{simulation_id}_complete_data.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(data_sources, f)
        
        self.logger.info(f"Exported pickle data to {filename}")
    
    def _export_hdf5(self, simulation_id: str, data_sources: Dict[str, Any]):
        """Export data in HDF5 format."""
        filename = self.output_dir / f"{simulation_id}_complete_data.h5"
        
        with h5py.File(filename, 'w') as f:
            for data_type, data in data_sources.items():
                if isinstance(data, list) and data:
                    # Convert to structured array for HDF5
                    df = pd.DataFrame(data)
                    for col in df.columns:
                        f.create_dataset(f"{data_type}/{col}", data=df[col].values)
        
        self.logger.info(f"Exported HDF5 data to {filename}")


class DataManagementLoggingToolkit:
    """Main interface for comprehensive data management and logging."""
    
    def __init__(self, output_dir: str, simulation_id: str):
        self.output_dir = Path(output_dir)
        self.simulation_id = simulation_id
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_output_dir = self.output_dir / f"{simulation_id}_{timestamp}"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.agent_logger = AgentStateLogger(str(self.run_output_dir / "agent_states"))
        self.llm_logger = LLMInteractionLogger(str(self.run_output_dir / "llm_interactions"))
        self.spatial_logger = SpatialDynamicsLogger(str(self.run_output_dir / "spatial_dynamics"))
        self.policy_logger = PolicyEffectLogger(str(self.run_output_dir / "policy_effects"))
        self.performance_monitor = PerformanceMonitor(str(self.run_output_dir / "performance"))
        self.export_manager = DataExportManager(str(self.run_output_dir / "exports"))
        
        # Simulation metadata
        self.simulation_metadata = {
            "simulation_id": simulation_id,
            "start_time": datetime.now().isoformat(),
            "output_directory": str(self.run_output_dir),
            "version": "1.0"
        }
        
        self.logger.info(f"Data Management Toolkit initialized for simulation {simulation_id}")
        self.logger.info(f"Output directory: {self.run_output_dir}")
    
    def log_simulation_step(self, time_step: int, model: Any, agents: List[Any],
                           persona_manager: Any, llm_interactions: List[Dict[str, Any]] = None,
                           policy_effects: List[Dict[str, Any]] = None,
                           performance_data: Dict[str, Any] = None):
        """Log all data for a single simulation step."""
        try:
            # Start performance timing
            self.performance_monitor.start_step_timing()
            
            # Log agent states
            for agent in agents:
                self.agent_logger.log_agent_state(self.simulation_id, time_step, agent, persona_manager)
            
            # Log LLM interactions
            if llm_interactions:
                for interaction in llm_interactions:
                    self.llm_logger.log_llm_interaction(
                        self.simulation_id, time_step, 
                        interaction.get('agent_id', 'unknown'), interaction
                    )
            
            # Log spatial state
            self.spatial_logger.log_spatial_state(self.simulation_id, time_step, model, agents)
            
            # Log policy effects
            if policy_effects:
                for effect_data in policy_effects:
                    policy = effect_data.get('policy')
                    impact_data = effect_data.get('impact_data', {})
                    affected_agents = effect_data.get('affected_agents', [])
                    
                    if policy:
                        self.policy_logger.log_policy_effect(
                            self.simulation_id, time_step, policy, impact_data, affected_agents
                        )
            
            # Log performance metrics
            if performance_data:
                self.performance_monitor.log_performance(self.simulation_id, time_step, performance_data)
            
            # Update simulation metadata
            self.simulation_metadata["last_step"] = time_step
            self.simulation_metadata["last_update"] = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"Error logging simulation step {time_step}: {e}")
    
    def finalize_simulation(self, export_formats: List[DataFormat] = None):
        """Finalize simulation logging and export data."""
        try:
            self.logger.info("Finalizing simulation data logging...")
            
            # Finalize all loggers
            self.agent_logger.finalize()
            self.llm_logger.finalize()
            
            # Export policy and performance data
            self.policy_logger.export_policy_data()
            self.performance_monitor.export_performance_data()
            
            # Collect all data for export
            data_sources = {
                "simulation_metadata": self.simulation_metadata,
                "policy_effects": [asdict(record) for record in self.policy_logger.policy_effects],
                "performance_metrics": [asdict(record) for record in self.performance_monitor.performance_records]
            }
            
            # Export in requested formats
            if export_formats:
                self.export_manager.export_simulation_data(self.simulation_id, data_sources, export_formats)
            
            # Save simulation metadata
            metadata_file = self.run_output_dir / "simulation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.simulation_metadata, f, indent=2)
            
            self.logger.info(f"Simulation data finalized and exported to {self.run_output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error finalizing simulation: {e}")
    
    def get_output_directory(self) -> str:
        """Get the output directory path."""
        return str(self.run_output_dir)
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get a summary of the simulation data."""
        return {
            "simulation_id": self.simulation_id,
            "output_directory": str(self.run_output_dir),
            "metadata": self.simulation_metadata,
            "agent_states_logged": len(self.agent_logger.state_buffer),
            "llm_interactions_logged": len(self.llm_logger.interaction_buffer),
            "policy_effects_logged": len(self.policy_logger.policy_effects),
            "performance_records": len(self.performance_monitor.performance_records)
        } 