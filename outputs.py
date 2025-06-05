"""
Output controller for the 15-minute city simulation.

This module handles various output metrics and statistics from the simulation,
including travel time tracking, agent behavior analysis, and summary reports.
"""

import logging
from typing import Dict, List, Any
import json
import os
from datetime import datetime


class OutputController:
    """
    Controls and manages various outputs from the simulation model.
    """
    
    def __init__(self, model):
        """
        Initialize the output controller.
        
        Args:
            model: The FifteenMinuteCity model instance
        """
        self.model = model
        self.logger = logging.getLogger("OutputController")
        
        # Initialize tracking variables
        self.total_travel_time = 0  # Total minutes spent traveling by all agents
        self.agent_travel_times = {}  # Individual agent travel times
        self.travel_events = []  # List of travel events for detailed analysis
        
        # New tracking variables for waiting and POI visits
        self.total_waiting_time = 0  # Total minutes spent waiting at POIs by all agents
        self.agent_waiting_times = {}  # Individual agent waiting times
        self.poi_visits_by_category = {  # Track POI visits by category
            "daily_living": 0,
            "healthcare": 0,
            "education": 0,
            "entertainment": 0,
            "transportation": 0
        }
        
        # Other potential metrics (for future expansion)
        self.poi_visits = {}  # Track POI visit counts
        self.energy_stats = {}  # Track energy consumption
        self.parish_mobility = {}  # Track inter-parish movement
        
    def track_travel_start(self, agent_id: int, from_node: int, to_node: int, travel_time: int):
        """
        Track when an agent starts traveling.
        
        Args:
            agent_id: ID of the traveling agent
            from_node: Starting node
            to_node: Destination node
            travel_time: Expected travel time in minutes
        """
        # Initialize agent travel time if not exists
        if agent_id not in self.agent_travel_times:
            self.agent_travel_times[agent_id] = 0
        
        # Add to total travel time
        self.total_travel_time += travel_time
        self.agent_travel_times[agent_id] += travel_time
        
        # Record travel event for detailed analysis
        travel_event = {
            'agent_id': agent_id,
            'step': self.model.step_count,
            'from_node': from_node,
            'to_node': to_node,
            'travel_time': travel_time,
            'timestamp': self.model.get_current_time()
        }
        self.travel_events.append(travel_event)
        
        self.logger.debug(f"Agent {agent_id} started travel: {travel_time} minutes from {from_node} to {to_node}")
    
    def track_waiting_start(self, agent_id: int, poi_category: str, waiting_time: int):
        """
        Track when an agent starts waiting at a POI.
        
        Args:
            agent_id: ID of the waiting agent
            poi_category: Category of the POI (daily_living, healthcare, etc.)
            waiting_time: Expected waiting time in minutes
        """
        # Initialize agent waiting time if not exists
        if agent_id not in self.agent_waiting_times:
            self.agent_waiting_times[agent_id] = 0
        
        # Add to total waiting time
        self.total_waiting_time += waiting_time
        self.agent_waiting_times[agent_id] += waiting_time
        
        self.logger.debug(f"Agent {agent_id} started waiting: {waiting_time} minutes at {poi_category} POI")
    
    def track_poi_visit(self, poi_category: str):
        """
        Track a POI visit by category.
        
        Args:
            poi_category: Category of the POI being visited
        """
        if poi_category in self.poi_visits_by_category:
            self.poi_visits_by_category[poi_category] += 1
        else:
            # Handle unknown categories
            if "other" not in self.poi_visits_by_category:
                self.poi_visits_by_category["other"] = 0
            self.poi_visits_by_category["other"] += 1
        
        self.logger.debug(f"POI visit tracked: {poi_category}")
    
    def get_total_travel_time(self) -> int:
        """
        Get the total travel time for all agents.
        
        Returns:
            Total travel time in minutes
        """
        return self.total_travel_time
    
    def get_agent_travel_times(self) -> Dict[int, int]:
        """
        Get travel times for individual agents.
        
        Returns:
            Dictionary mapping agent IDs to their total travel time
        """
        return self.agent_travel_times.copy()
    
    def get_average_travel_time_per_agent(self) -> float:
        """
        Calculate the average travel time per agent.
        
        Returns:
            Average travel time in minutes
        """
        if not self.agent_travel_times:
            return 0.0
        return self.total_travel_time / len(self.agent_travel_times)
    
    def get_total_waiting_time(self) -> int:
        """
        Get the total waiting time for all agents.
        
        Returns:
            Total waiting time in minutes
        """
        return self.total_waiting_time
    
    def get_poi_visits_by_category(self) -> Dict[str, int]:
        """
        Get POI visits by category.
        
        Returns:
            Dictionary mapping POI categories to visit counts
        """
        return self.poi_visits_by_category.copy()
    
    def print_travel_summary(self):
        """
        Print a summary of travel statistics.
        """
        print("\n" + "="*60)
        print("SIMULATION TRAVEL SUMMARY")
        print("="*60)
        
        # Basic travel statistics
        print(f"Total travel time (all agents): {self.total_travel_time} minutes")
        print(f"Total travel time (all agents): {self.total_travel_time / 60:.1f} hours")
        
        if self.agent_travel_times:
            print(f"Number of agents that traveled: {len(self.agent_travel_times)}")
            print(f"Average travel time per agent: {self.get_average_travel_time_per_agent():.1f} minutes")
            
            # Find agent with most and least travel
            max_agent = max(self.agent_travel_times.items(), key=lambda x: x[1])
            min_agent = min(self.agent_travel_times.items(), key=lambda x: x[1])
            
            print(f"Most traveled agent: Resident {max_agent[0]} ({max_agent[1]} minutes)")
            print(f"Least traveled agent: Resident {min_agent[0]} ({min_agent[1]} minutes)")
        else:
            print("No agents traveled during the simulation.")
        
        # Travel events summary
        print(f"Total travel events: {len(self.travel_events)}")
        
        if self.travel_events:
            avg_trip_length = sum(event['travel_time'] for event in self.travel_events) / len(self.travel_events)
            print(f"Average trip length: {avg_trip_length:.1f} minutes")
        
        # Waiting time statistics
        print(f"\nTotal waiting time (all agents): {self.total_waiting_time} minutes")
        print(f"Total waiting time (all agents): {self.total_waiting_time / 60:.1f} hours")
        
        if self.agent_waiting_times:
            print(f"Number of agents that waited: {len(self.agent_waiting_times)}")
            avg_waiting_per_agent = self.total_waiting_time / len(self.agent_waiting_times)
            print(f"Average waiting time per agent: {avg_waiting_per_agent:.1f} minutes")
        else:
            print("No agents waited at POIs during the simulation.")
        
        # POI visit statistics
        print(f"\nPOI visits by category:")
        total_visits = sum(self.poi_visits_by_category.values())
        print(f"Total POI visits: {total_visits}")
        
        for category, count in self.poi_visits_by_category.items():
            if count > 0:
                percentage = (count / total_visits * 100) if total_visits > 0 else 0
                print(f"  - {category.replace('_', ' ').title()}: {count} visits ({percentage:.1f}%)")
        
        print("="*60)
    
    def save_detailed_report(self, filepath: str = None):
        """
        Save a detailed report to a JSON file.
        
        Args:
            filepath: Path to save the report. If None, uses default naming.
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"simulation_report_{timestamp}.json"
        
        report = {
            'simulation_info': {
                'total_steps': self.model.step_count,
                'total_agents': len(self.model.residents),
                'simulation_time': self.model.get_current_time()
            },
            'travel_statistics': {
                'total_travel_time_minutes': self.total_travel_time,
                'total_travel_time_hours': self.total_travel_time / 60,
                'average_travel_time_per_agent': self.get_average_travel_time_per_agent(),
                'agent_travel_times': self.agent_travel_times,
                'total_travel_events': len(self.travel_events)
            },
            'waiting_statistics': {
                'total_waiting_time_minutes': self.total_waiting_time,
                'total_waiting_time_hours': self.total_waiting_time / 60,
                'agent_waiting_times': self.agent_waiting_times,
                'number_of_agents_that_waited': len(self.agent_waiting_times)
            },
            'poi_visit_statistics': {
                'visits_by_category': self.poi_visits_by_category,
                'total_visits': sum(self.poi_visits_by_category.values())
            },
            'travel_events': self.travel_events
        }
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"\nDetailed report saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    def reset(self):
        """
        Reset all tracking variables (useful for multiple simulation runs).
        """
        self.total_travel_time = 0
        self.agent_travel_times = {}
        self.travel_events = []
        self.total_waiting_time = 0
        self.agent_waiting_times = {}
        self.poi_visits_by_category = {
            "daily_living": 0,
            "healthcare": 0,
            "education": 0,
            "entertainment": 0,
            "transportation": 0
        }
        self.poi_visits = {}
        self.energy_stats = {}
        self.parish_mobility = {}
        
        self.logger.info("Output controller reset")


# Convenience function for easy access
def create_output_controller(model):
    """
    Create and return an OutputController instance.
    
    Args:
        model: The FifteenMinuteCity model instance
        
    Returns:
        OutputController instance
    """
    return OutputController(model) 