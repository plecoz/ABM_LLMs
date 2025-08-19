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
import uuid


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
            "transportation": 0,
            "casino": 0
        }
        
        # TEMPORARY: Track tourist healthcare visits
        self.tourist_healthcare_visits = set()  # Set of tourist IDs who visited healthcare POIs
        
        # Other potential metrics (for future expansion)
        self.poi_visits = {}  # Track POI visit counts
        self.energy_stats = {}  # Track energy consumption
        self.parish_mobility = {}  # Track inter-parish movement
        
    def track_travel_step(self, agent_id: int):
        """
        Track one step of actual travel time for an agent.
        
        Args:
            agent_id: ID of the traveling agent
        """
        # Initialize agent travel time if not exists
        if agent_id not in self.agent_travel_times:
            self.agent_travel_times[agent_id] = 0
        
        # Add 1 minute of actual travel time
        self.total_travel_time += 1
        self.agent_travel_times[agent_id] += 1
        
        # Update the most recent travel event for this agent with actual travel time
        for event in reversed(self.travel_events):
            if event['agent_id'] == agent_id:
                event['travel_time'] += 1
                break
        
        self.logger.debug(f"Agent {agent_id} traveled for 1 minute (total: {self.agent_travel_times[agent_id]})")
    
    def track_travel_start(self, agent_id: int, from_node: int, to_node: int, travel_time: int):
        """
        Track when an agent starts traveling (for event logging only, not time accumulation).
        
        Args:
            agent_id: ID of the traveling agent
            from_node: Starting node ID
            to_node: Destination node ID
            travel_time: Planned travel time in minutes (logged but not added to statistics)
        """
        # Store travel event for detailed analysis (but don't add planned time to statistics)
        # The actual travel time will be tracked by track_travel_step() calls
        travel_event = {
            'agent_id': agent_id,
            'step': self.model.step_count,
            'from_node': from_node,
            'to_node': to_node,
            'travel_time': 0,  # Actual travel time starts at 0, will be updated by track_travel_step
            'timestamp': self.model.get_current_time()
        }
        self.travel_events.append(travel_event)
        
        # self.logger.info(f"Agent {agent_id} started traveling from node {from_node} to node {to_node} "
        #                 f"(planned time: {travel_time} minutes)")
        
        # Initialize agent travel time tracking if not exists (for when they actually start traveling)
        if agent_id not in self.agent_travel_times:
            self.agent_travel_times[agent_id] = 0
    
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
    
    def track_poi_visit(self, poi_category: str, agent_id: int = None):
        """
        Track when an agent visits a POI.
        
        Args:
            poi_category: Category of the POI visited
            agent_id: ID of the agent visiting (optional, for tourist tracking)
        """
        if poi_category in self.poi_visits_by_category:
            self.poi_visits_by_category[poi_category] += 1
        else:
            # Handle new categories that might not be in our initial list
            self.poi_visits_by_category[poi_category] = 1
        
        # TEMPORARY: Track tourist healthcare visits
        if poi_category == "healthcare" and agent_id is not None:
            # Check if the agent is a tourist
            agent = self.model.get_agent_by_id(agent_id)
            if agent and hasattr(agent, 'is_tourist') and agent.is_tourist:
                self.tourist_healthcare_visits.add(agent_id)
        
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
        Get the number of POI visits by category.
        
        Returns:
            Dictionary mapping POI categories to visit counts
        """
        return self.poi_visits_by_category.copy()
    
    def get_tourist_healthcare_percentage(self) -> Dict[str, float]:
        """
        TEMPORARY: Calculate percentage of tourists who accessed/didn't access healthcare POIs.
        
        Returns:
            Dictionary with tourist healthcare access statistics
        """
        # Count total tourists in the simulation
        total_tourists = 0
        for agent in self.model.residents:
            if hasattr(agent, 'is_tourist') and agent.is_tourist:
                total_tourists += 1
        
        if total_tourists == 0:
            return {
                "total_tourists": 0,
                "tourists_accessed_healthcare": 0,
                "tourists_no_healthcare": 0,
                "percentage_accessed": 0.0,
                "percentage_no_access": 0.0
            }
        
        tourists_accessed = len(self.tourist_healthcare_visits)
        tourists_no_access = total_tourists - tourists_accessed
        
        percentage_accessed = (tourists_accessed / total_tourists) * 100
        percentage_no_access = (tourists_no_access / total_tourists) * 100
        
        return {
            "total_tourists": total_tourists,
            "tourists_accessed_healthcare": tourists_accessed,
            "tourists_no_healthcare": tourists_no_access,
            "percentage_accessed": percentage_accessed,
            "percentage_no_access": percentage_no_access
        }
    
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
        total_poi_visits = sum(self.poi_visits_by_category.values())
        print(f"Total POI visits: {total_poi_visits}")
        
        for category, count in self.poi_visits_by_category.items():
            if count > 0:
                print(f"  - {category.replace('_', ' ').title()}: {count}")
        
        # TEMPORARY: Print tourist healthcare statistics
        tourist_stats = self.get_tourist_healthcare_percentage()
        if tourist_stats["total_tourists"] > 0:
            print(f"\n--- TOURIST HEALTHCARE ACCESS ---")
            print(f"Total tourists: {tourist_stats['total_tourists']}")
            print(f"Tourists who accessed healthcare: {tourist_stats['tourists_accessed_healthcare']}")
            print(f"Tourists who didn't access healthcare: {tourist_stats['tourists_no_healthcare']}")
            print(f"Percentage who accessed healthcare: {tourist_stats['percentage_accessed']:.1f}%")
            print(f"Percentage who didn't access healthcare: {tourist_stats['percentage_no_access']:.1f}%")
        
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

    def save_summary_report(self, filepath: str = None):
        """
        Save a concise JSON summary of the simulation.

        The summary includes:
        - simulation_id
        - city
        - total_steps and current time string
        - number of residents
        - list of parishes present among residents
        - temperature info (base/current and model params)
        - POI visit totals (by category and overall)
        - total travel time (and average per agent)
        - unmet demand by parish (POIs that couldn't be accessed)

        Args:
            filepath: Target path. If provided and ends with .json, a sibling
                      file with suffix _summary.json will be written.
                      If None, writes to summary_report_<timestamp>.json
        """
        # Build simulation id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        simulation_id = f"sim-{timestamp}-{uuid.uuid4().hex[:8]}"

        # Core counts
        num_residents = len(getattr(self.model, 'residents', []))

        # Parishes present among residents
        parishes = sorted({getattr(a, 'parish', None) for a in getattr(self.model, 'residents', []) if getattr(a, 'parish', None)})

        # Temperature info
        temp_info = {}
        if hasattr(self.model, 'get_temperature_info'):
            temp_info = self.model.get_temperature_info()
        # add time period and hour for context
        temp_info.update({
            'time_period': getattr(self.model, 'time_period', None),
            'hour': getattr(self.model, 'hour_of_day', None)
        })

        # POI visits
        visits_by_category = self.get_poi_visits_by_category()
        total_poi_visits = sum(visits_by_category.values()) if visits_by_category else 0

        # Travel stats
        travel_summary = {
            'total_travel_time_minutes': self.total_travel_time,
            'average_travel_time_per_agent': round(self.get_average_travel_time_per_agent(), 2),
            'total_travel_events': len(self.travel_events)
        }

        # Health status summary (current state at end of simulation)
        health_counts: Dict[str, int] = {}
        for a in getattr(self.model, 'residents', []):
            status = getattr(a, 'health_status', None) or 'unknown'
            health_counts[status] = health_counts.get(status, 0) + 1
        num_sick = health_counts.get('sick', 0)
        percent_sick = (num_sick / num_residents * 100.0) if num_residents else 0.0
        health_summary = {
            'num_residents': num_residents,
            'num_sick': num_sick,
            'percent_sick': round(percent_sick, 2),
            'by_status': health_counts,
        }

        # Unmet demand (POIs that couldn't be accessed)
        unmet = getattr(self.model, 'unmet_demand_by_parish', {}) or {}

        # Compose summary
        summary = {
            'simulation_id': simulation_id,
            'city': getattr(self.model, 'city', None),
            'total_steps': getattr(self.model, 'step_count', 0),
            'current_time_string': (self.model.get_current_time().get('time_string')
                                     if hasattr(self.model, 'get_current_time') else None),
            'num_residents': num_residents,
            'parishes': parishes,
            'temperature': temp_info,
            'poi_visits': {
                'total': total_poi_visits,
                'by_category': visits_by_category
            },
            'travel': travel_summary,
            'health': health_summary,
            'unmet_demand_by_parish': unmet,
            'healthcare_access': {
                'per_resident': [
                    {
                        'agent_id': getattr(a, 'unique_id', None),
                        'home_node': getattr(a, 'home_node', None),
                        'parish': getattr(a, 'parish', None),
                        'success': bool(getattr(a, 'healthcare_access_success', False))
                    }
                    for a in getattr(self.model, 'residents', [])
                ],
                'summary': {
                    'num_success': sum(1 for a in getattr(self.model, 'residents', []) if getattr(a, 'healthcare_access_success', False)),
                    'num_total': num_residents
                }
            }
        }

        # Determine output path
        if filepath:
            base, ext = os.path.splitext(filepath)
            out_path = f"{base}_summary.json" if ext.lower() == '.json' else filepath
        else:
            out_path = f"summary_report_{timestamp}.json"

        try:
            os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\nSummary report saved to: {out_path}")
        except Exception as e:
            self.logger.error(f"Failed to save summary report: {e}")

    def save_healthcare_access_points(self, filepath: str):
        """
        Save a minimal JSON file for healthcare accessibility mapping.

        Contents per resident:
        - agent_id
        - home_node
        - x, y (node coordinates)
        - success (bool) — True if resident completed visit_doctor or go_pharmacy at least once
        """
        records = []
        graph = getattr(self.model, 'graph', None)

        for a in getattr(self.model, 'residents', []):
            node_id = getattr(a, 'home_node', None)
            x = y = None
            if graph is not None and node_id is not None and node_id in graph.nodes:
                node_data = graph.nodes[node_id]
                x = node_data.get('x')
                y = node_data.get('y')
            rec = {
                'agent_id': getattr(a, 'unique_id', None),
                'home_node': node_id,
                'x': x,
                'y': y,
                'success': getattr(a, 'healthcare_access_success', None)
            }
            records.append(rec)

        payload = {
            'simulation_id': f"hc-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'city': getattr(self.model, 'city', None),
            'points': records
        }

        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f"Healthcare access points saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save healthcare access points: {e}")
    
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
            "transportation": 0,
            "casino": 0
        }
        self.poi_visits = {}
        self.energy_stats = {}
        self.parish_mobility = {}
        self.tourist_healthcare_visits = set()
        
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