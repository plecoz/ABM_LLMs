#!/usr/bin/env python3
"""
Configuration module for Data Management & Logging Toolkit

This module provides configuration classes and utilities for customizing
the logging behavior, data formats, performance settings, and analysis
parameters of the ABM simulation data management system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import json
from pathlib import Path

from simulation.data_management_logging import DataFormat, LogLevel


class LoggingMode(Enum):
    """Logging modes for different use cases."""
    MINIMAL = "minimal"          # Essential data only
    STANDARD = "standard"        # Standard research logging
    COMPREHENSIVE = "comprehensive"  # Full debugging and analysis
    CUSTOM = "custom"           # User-defined configuration


class PerformanceLevel(Enum):
    """Performance optimization levels."""
    HIGH_PERFORMANCE = "high_performance"  # Minimal logging for speed
    BALANCED = "balanced"                  # Balance between logging and performance
    DETAILED = "detailed"                  # Comprehensive logging, slower


@dataclass
class AgentLoggingConfig:
    """Configuration for agent state logging."""
    enabled: bool = True
    buffer_size: int = 1000
    log_frequency: int = 1  # Log every N steps
    
    # What to log
    log_basic_attributes: bool = True
    log_emotional_state: bool = True
    log_decision_history: bool = True
    log_policy_contexts: bool = True
    log_spatial_data: bool = True
    
    # Performance settings
    compress_data: bool = True
    use_binary_format: bool = False
    
    # Filtering
    persona_types_filter: Optional[List[str]] = None  # Log only specific personas
    attribute_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMLoggingConfig:
    """Configuration for LLM interaction logging."""
    enabled: bool = True
    buffer_size: int = 100
    
    # What to log
    log_prompts: bool = True
    log_responses: bool = True
    log_agent_context: bool = True
    log_performance_metrics: bool = True
    log_quality_scores: bool = True
    
    # Privacy and security
    anonymize_prompts: bool = False
    truncate_long_responses: int = 0  # 0 = no truncation
    
    # Performance analysis
    calculate_quality_scores: bool = True
    track_token_usage: bool = True
    estimate_costs: bool = True
    
    # Database settings
    use_sqlite: bool = True
    create_indexes: bool = True


@dataclass
class SpatialLoggingConfig:
    """Configuration for spatial dynamics logging."""
    enabled: bool = True
    log_frequency: int = 1
    
    # What to log
    log_agent_positions: bool = True
    log_facility_status: bool = True
    log_environmental_state: bool = True
    log_interactions: bool = True
    
    # Data format
    use_hdf5: bool = True
    compress_spatial_data: bool = True
    
    # Spatial resolution
    position_precision: int = 6  # Decimal places for coordinates
    track_movement_history: bool = False


@dataclass
class PolicyLoggingConfig:
    """Configuration for policy effect logging."""
    enabled: bool = True
    
    # What to log
    log_implementation_details: bool = True
    log_agent_responses: bool = True
    log_impact_metrics: bool = True
    log_acceptance_rates: bool = True
    
    # Analysis settings
    calculate_baseline_metrics: bool = True
    track_trend_analysis: bool = True
    
    # Export settings
    export_formats: List[DataFormat] = field(default_factory=lambda: [DataFormat.JSON, DataFormat.CSV])


@dataclass
class PerformanceLoggingConfig:
    """Configuration for performance monitoring."""
    enabled: bool = True
    
    # What to monitor
    track_step_timing: bool = True
    track_memory_usage: bool = True
    track_cpu_usage: bool = True
    track_llm_performance: bool = True
    
    # System monitoring
    monitor_system_resources: bool = False  # Requires psutil
    detailed_profiling: bool = False  # Detailed code profiling
    
    # Reporting
    generate_performance_reports: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'max_step_duration_ms': 10000,
        'max_memory_usage_mb': 2000,
        'max_cpu_usage_percent': 90
    })


@dataclass
class ExportConfig:
    """Configuration for data export."""
    auto_export: bool = True
    export_formats: List[DataFormat] = field(default_factory=lambda: [DataFormat.JSON, DataFormat.CSV])
    
    # Compression
    compress_exports: bool = True
    compression_level: int = 6  # 1-9 for gzip
    
    # File organization
    organize_by_date: bool = True
    organize_by_simulation_id: bool = True
    
    # Export frequency
    export_frequency: int = 0  # 0 = only at end, N = every N steps
    incremental_export: bool = False


@dataclass
class DataManagementConfig:
    """Main configuration class for Data Management & Logging Toolkit."""
    
    # General settings
    simulation_id: str = "abm_simulation"
    output_directory: str = "simulation_data"
    logging_mode: LoggingMode = LoggingMode.STANDARD
    performance_level: PerformanceLevel = PerformanceLevel.BALANCED
    
    # Component configurations
    agent_logging: AgentLoggingConfig = field(default_factory=AgentLoggingConfig)
    llm_logging: LLMLoggingConfig = field(default_factory=LLMLoggingConfig)
    spatial_logging: SpatialLoggingConfig = field(default_factory=SpatialLoggingConfig)
    policy_logging: PolicyLoggingConfig = field(default_factory=PolicyLoggingConfig)
    performance_logging: PerformanceLoggingConfig = field(default_factory=PerformanceLoggingConfig)
    export_config: ExportConfig = field(default_factory=ExportConfig)
    
    # Advanced settings
    debug_mode: bool = False
    verbose_logging: bool = False
    log_level: LogLevel = LogLevel.INFO
    
    # Data retention
    max_log_files: int = 100
    max_storage_gb: float = 10.0
    auto_cleanup: bool = True
    
    @classmethod
    def create_minimal_config(cls) -> 'DataManagementConfig':
        """Create a minimal logging configuration for high performance."""
        config = cls()
        config.logging_mode = LoggingMode.MINIMAL
        config.performance_level = PerformanceLevel.HIGH_PERFORMANCE
        
        # Minimal agent logging
        config.agent_logging.log_frequency = 5
        config.agent_logging.log_emotional_state = False
        config.agent_logging.log_decision_history = False
        config.agent_logging.buffer_size = 2000
        
        # Minimal LLM logging
        config.llm_logging.log_agent_context = False
        config.llm_logging.log_quality_scores = False
        config.llm_logging.calculate_quality_scores = False
        
        # Minimal spatial logging
        config.spatial_logging.log_frequency = 10
        config.spatial_logging.log_interactions = False
        config.spatial_logging.track_movement_history = False
        
        # Minimal performance logging
        config.performance_logging.track_memory_usage = False
        config.performance_logging.track_cpu_usage = False
        config.performance_logging.detailed_profiling = False
        
        return config
    
    @classmethod
    def create_comprehensive_config(cls) -> 'DataManagementConfig':
        """Create a comprehensive logging configuration for detailed analysis."""
        config = cls()
        config.logging_mode = LoggingMode.COMPREHENSIVE
        config.performance_level = PerformanceLevel.DETAILED
        config.debug_mode = True
        config.verbose_logging = True
        
        # Comprehensive agent logging
        config.agent_logging.buffer_size = 500  # Smaller buffer for more frequent writes
        config.agent_logging.log_frequency = 1
        config.agent_logging.use_binary_format = True
        
        # Comprehensive LLM logging
        config.llm_logging.calculate_quality_scores = True
        config.llm_logging.track_token_usage = True
        config.llm_logging.estimate_costs = True
        
        # Comprehensive spatial logging
        config.spatial_logging.track_movement_history = True
        config.spatial_logging.position_precision = 8
        
        # Comprehensive policy logging
        config.policy_logging.track_trend_analysis = True
        config.policy_logging.export_formats = [
            DataFormat.JSON, DataFormat.CSV, DataFormat.PICKLE, DataFormat.HDF5
        ]
        
        # Comprehensive performance logging
        config.performance_logging.monitor_system_resources = True
        config.performance_logging.detailed_profiling = True
        
        # Comprehensive export
        config.export_config.export_formats = [
            DataFormat.JSON, DataFormat.CSV, DataFormat.PICKLE, DataFormat.HDF5
        ]
        config.export_config.incremental_export = True
        config.export_config.export_frequency = 10
        
        return config
    
    @classmethod
    def create_research_config(cls) -> 'DataManagementConfig':
        """Create a research-focused configuration balancing detail and performance."""
        config = cls()
        config.logging_mode = LoggingMode.STANDARD
        config.performance_level = PerformanceLevel.BALANCED
        
        # Research-focused settings
        config.agent_logging.log_frequency = 1
        config.llm_logging.calculate_quality_scores = True
        config.spatial_logging.log_frequency = 2
        config.policy_logging.track_trend_analysis = True
        config.performance_logging.generate_performance_reports = True
        
        # Export for analysis
        config.export_config.export_formats = [DataFormat.JSON, DataFormat.CSV, DataFormat.PARQUET]
        config.export_config.organize_by_date = True
        
        return config
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = self._to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'DataManagementConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls._from_dict(config_dict)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'simulation_id': self.simulation_id,
            'output_directory': self.output_directory,
            'logging_mode': self.logging_mode.value,
            'performance_level': self.performance_level.value,
            'agent_logging': {
                'enabled': self.agent_logging.enabled,
                'buffer_size': self.agent_logging.buffer_size,
                'log_frequency': self.agent_logging.log_frequency,
                'log_basic_attributes': self.agent_logging.log_basic_attributes,
                'log_emotional_state': self.agent_logging.log_emotional_state,
                'log_decision_history': self.agent_logging.log_decision_history,
                'log_policy_contexts': self.agent_logging.log_policy_contexts,
                'log_spatial_data': self.agent_logging.log_spatial_data,
                'compress_data': self.agent_logging.compress_data,
                'use_binary_format': self.agent_logging.use_binary_format,
                'persona_types_filter': self.agent_logging.persona_types_filter,
                'attribute_filters': self.agent_logging.attribute_filters
            },
            'llm_logging': {
                'enabled': self.llm_logging.enabled,
                'buffer_size': self.llm_logging.buffer_size,
                'log_prompts': self.llm_logging.log_prompts,
                'log_responses': self.llm_logging.log_responses,
                'log_agent_context': self.llm_logging.log_agent_context,
                'log_performance_metrics': self.llm_logging.log_performance_metrics,
                'log_quality_scores': self.llm_logging.log_quality_scores,
                'anonymize_prompts': self.llm_logging.anonymize_prompts,
                'truncate_long_responses': self.llm_logging.truncate_long_responses,
                'calculate_quality_scores': self.llm_logging.calculate_quality_scores,
                'track_token_usage': self.llm_logging.track_token_usage,
                'estimate_costs': self.llm_logging.estimate_costs,
                'use_sqlite': self.llm_logging.use_sqlite,
                'create_indexes': self.llm_logging.create_indexes
            },
            'spatial_logging': {
                'enabled': self.spatial_logging.enabled,
                'log_frequency': self.spatial_logging.log_frequency,
                'log_agent_positions': self.spatial_logging.log_agent_positions,
                'log_facility_status': self.spatial_logging.log_facility_status,
                'log_environmental_state': self.spatial_logging.log_environmental_state,
                'log_interactions': self.spatial_logging.log_interactions,
                'use_hdf5': self.spatial_logging.use_hdf5,
                'compress_spatial_data': self.spatial_logging.compress_spatial_data,
                'position_precision': self.spatial_logging.position_precision,
                'track_movement_history': self.spatial_logging.track_movement_history
            },
            'policy_logging': {
                'enabled': self.policy_logging.enabled,
                'log_implementation_details': self.policy_logging.log_implementation_details,
                'log_agent_responses': self.policy_logging.log_agent_responses,
                'log_impact_metrics': self.policy_logging.log_impact_metrics,
                'log_acceptance_rates': self.policy_logging.log_acceptance_rates,
                'calculate_baseline_metrics': self.policy_logging.calculate_baseline_metrics,
                'track_trend_analysis': self.policy_logging.track_trend_analysis,
                'export_formats': [fmt.value for fmt in self.policy_logging.export_formats]
            },
            'performance_logging': {
                'enabled': self.performance_logging.enabled,
                'track_step_timing': self.performance_logging.track_step_timing,
                'track_memory_usage': self.performance_logging.track_memory_usage,
                'track_cpu_usage': self.performance_logging.track_cpu_usage,
                'track_llm_performance': self.performance_logging.track_llm_performance,
                'monitor_system_resources': self.performance_logging.monitor_system_resources,
                'detailed_profiling': self.performance_logging.detailed_profiling,
                'generate_performance_reports': self.performance_logging.generate_performance_reports,
                'alert_thresholds': self.performance_logging.alert_thresholds
            },
            'export_config': {
                'auto_export': self.export_config.auto_export,
                'export_formats': [fmt.value for fmt in self.export_config.export_formats],
                'compress_exports': self.export_config.compress_exports,
                'compression_level': self.export_config.compression_level,
                'organize_by_date': self.export_config.organize_by_date,
                'organize_by_simulation_id': self.export_config.organize_by_simulation_id,
                'export_frequency': self.export_config.export_frequency,
                'incremental_export': self.export_config.incremental_export
            },
            'debug_mode': self.debug_mode,
            'verbose_logging': self.verbose_logging,
            'log_level': self.log_level.value,
            'max_log_files': self.max_log_files,
            'max_storage_gb': self.max_storage_gb,
            'auto_cleanup': self.auto_cleanup
        }
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> 'DataManagementConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Basic settings
        config.simulation_id = config_dict.get('simulation_id', config.simulation_id)
        config.output_directory = config_dict.get('output_directory', config.output_directory)
        config.logging_mode = LoggingMode(config_dict.get('logging_mode', config.logging_mode.value))
        config.performance_level = PerformanceLevel(config_dict.get('performance_level', config.performance_level.value))
        
        # Component configurations
        if 'agent_logging' in config_dict:
            al = config_dict['agent_logging']
            config.agent_logging = AgentLoggingConfig(
                enabled=al.get('enabled', True),
                buffer_size=al.get('buffer_size', 1000),
                log_frequency=al.get('log_frequency', 1),
                log_basic_attributes=al.get('log_basic_attributes', True),
                log_emotional_state=al.get('log_emotional_state', True),
                log_decision_history=al.get('log_decision_history', True),
                log_policy_contexts=al.get('log_policy_contexts', True),
                log_spatial_data=al.get('log_spatial_data', True),
                compress_data=al.get('compress_data', True),
                use_binary_format=al.get('use_binary_format', False),
                persona_types_filter=al.get('persona_types_filter'),
                attribute_filters=al.get('attribute_filters', {})
            )
        
        # Similar for other components...
        # (Implementation continues for all config components)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []
        
        # Check buffer sizes
        if self.agent_logging.buffer_size < 100:
            warnings.append("Agent logging buffer size is very small, may impact performance")
        
        if self.llm_logging.buffer_size < 10:
            warnings.append("LLM logging buffer size is very small, may impact performance")
        
        # Check storage settings
        if self.max_storage_gb < 1.0:
            warnings.append("Maximum storage limit is very low, may cause data loss")
        
        # Check export formats
        if not self.export_config.export_formats:
            warnings.append("No export formats specified, data may not be accessible")
        
        # Check performance settings
        if (self.performance_level == PerformanceLevel.HIGH_PERFORMANCE and 
            self.logging_mode == LoggingMode.COMPREHENSIVE):
            warnings.append("High performance mode with comprehensive logging may cause conflicts")
        
        return warnings


def create_default_configs() -> Dict[str, DataManagementConfig]:
    """Create default configurations for common use cases."""
    return {
        'minimal': DataManagementConfig.create_minimal_config(),
        'standard': DataManagementConfig.create_research_config(),
        'comprehensive': DataManagementConfig.create_comprehensive_config(),
        'research': DataManagementConfig.create_research_config()
    }


def save_default_configs(output_dir: str = "config"):
    """Save default configurations to files."""
    configs = create_default_configs()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for name, config in configs.items():
        config.save_to_file(str(output_path / f"logging_config_{name}.json"))
    
    print(f"Saved default configurations to {output_dir}/")


if __name__ == "__main__":
    # Create and save default configurations
    save_default_configs()
    
    # Example usage
    print("Data Management & Logging Configuration Examples:")
    print("=" * 50)
    
    configs = create_default_configs()
    
    for name, config in configs.items():
        print(f"\n{name.upper()} Configuration:")
        print(f"  Logging Mode: {config.logging_mode.value}")
        print(f"  Performance Level: {config.performance_level.value}")
        print(f"  Agent Logging Frequency: {config.agent_logging.log_frequency}")
        print(f"  LLM Quality Scoring: {config.llm_logging.calculate_quality_scores}")
        print(f"  Export Formats: {[fmt.value for fmt in config.export_config.export_formats]}")
        
        warnings = config.validate()
        if warnings:
            print(f"  Warnings: {len(warnings)}")
            for warning in warnings:
                print(f"    - {warning}")
        else:
            print(f"  Status: ✅ Valid configuration") 