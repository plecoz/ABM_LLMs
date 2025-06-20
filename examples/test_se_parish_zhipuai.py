#!/usr/bin/env python3
"""
Test script for Sé parish simulation with 10 residents using ZhipuAI LLMs.

This script demonstrates:
1. Setting up 10 residents in Sé parish (written as "S" in the data)
2. Using ZhipuAI's glm-4-flash model for decision-making
3. LLM-driven POI selection based on resident needs and preferences
4. Comprehensive logging and data collection
"""

import os
import sys
import json
import pickle
import random
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.llm_api import LLMWrapper
from simulation.fifteenminutescity_model import FifteenMinuteCity
from simulation.llm_interaction_layer_fifteenminutescity import FifteenMinuteCityLLMLayer
from simulation.data_management_logging import DataManagementLoggingToolkit
from simulation.logging_config import DataManagementConfig, LoggingMode
from agents.fifteenminutescity.resident import Resident
from agents.fifteenminutescity.persona_memory_modules import PersonaType
from environment.fifteenminutescity.city_network import get_or_load_city_network
from environment.fifteenminutescity.pois import get_or_fetch_pois, get_or_fetch_environment_data
import geopandas as gpd

class SeParishTestRunner:
    """Test runner for Sé parish simulation with ZhipuAI LLMs."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.city = "Macau, China"
        self.target_parish = "Coloane"  # Coloane parish 
        self.num_residents = 10
        self.simulation_steps = 100  # 100 minutes of simulation
        self.zhipuai_model = "glm-4-flash"
        self.zhipuai_platform = "zhipuai"
        
        # Initialize components
        self.llm_wrapper = None
        self.graph = None
        self.pois = None
        self.parishes_gdf = None
        self.model = None
        self.data_toolkit = None
        
        # Results storage
        self.results = {
            "start_time": None,
            "end_time": None,
            "residents": [],
            "poi_visits": [],
            "llm_decisions": [],
            "performance_metrics": {}
        }
        
    def setup_zhipuai_llm(self):
        """Set up ZhipuAI LLM wrapper."""
        print("Setting up ZhipuAI LLM...")
        try:
            # Set the API key as environment variable
            os.environ["ZHIPUAI_API_KEY"] = "8a8922a87b9c404c86a33c9f84890f53.h1orz28xqPNUjLpl"
            
            self.llm_wrapper = LLMWrapper(
                model_name=self.zhipuai_model,
                platform=self.zhipuai_platform
            )
            
            # Test the connection
            test_response = self.llm_wrapper.get_response("Hello, are you working?")
            print(f"✓ ZhipuAI LLM connected successfully!")
            print(f"  Model: {self.zhipuai_model}")
            print(f"  Test response: {test_response[:100]}...")
            
            return True
        except Exception as e:
            print(f"✗ Failed to connect to ZhipuAI LLM: {e}")
            print("  Continuing with mock LLM responses...")
            return False
    
    def load_city_data(self):
        """Load city network, POIs, and parish data."""
        print(f"Loading city data for {self.city}...")
        
        # Load street network
        print("- Loading street network...")
        self.graph = get_or_load_city_network(
            place_name=self.city,
            mode="walk",
            load_path="data/macau_shapefiles/macau_network.pkl"
        )
        print(f"✓ Loaded {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
        
        # Load POIs
        print("- Loading POIs...")
        self.pois = get_or_fetch_pois(
            graph=self.graph,
            place_name=self.city,
            load_path="data/macau_shapefiles/macau_pois.pkl"
        )
        
        # Count POIs
        total_pois = sum(len(poi_list) for poi_list in self.pois.values())
        print(f"✓ Loaded {total_pois} POIs across {len(self.pois)} categories")
        for category, poi_list in self.pois.items():
            if poi_list:
                print(f"  - {category}: {len(poi_list)} POIs")
        
        # Load parishes
        print("- Loading parish data...")
        try:
            self.parishes_gdf = gpd.read_file("data/macau_shapefiles/macau_new_districts.gpkg")
            print(f"✓ Loaded {len(self.parishes_gdf)} parishes")
            
            # Check if Sé parish exists
            parish_names = self.parishes_gdf['name'].tolist()
            if self.target_parish in parish_names:
                print(f"✓ Found target parish: {self.target_parish}")
            else:
                print(f"⚠ Target parish '{self.target_parish}' not found in: {parish_names}")
        except Exception as e:
            print(f"⚠ Could not load parish data: {e}")
            self.parishes_gdf = None
    
    def filter_to_se_parish(self):
        """Filter the graph and POIs to only include Sé parish."""
        if self.parishes_gdf is None:
            print("⚠ No parish data available, using full city")
            return
        
        print(f"Filtering to {self.target_parish} parish...")
        
        # Find Sé parish geometry
        se_parish = self.parishes_gdf[self.parishes_gdf['name'] == self.target_parish]
        if se_parish.empty:
            print(f"⚠ Parish '{self.target_parish}' not found, using full city")
            return
        
        se_geometry = se_parish.iloc[0]['geometry']
        
        # Filter nodes to those within Sé parish
        from shapely.geometry import Point
        nodes_in_se = []
        
        for node_id, node_attrs in self.graph.nodes(data=True):
            if 'x' in node_attrs and 'y' in node_attrs:
                point = Point(node_attrs['x'], node_attrs['y'])
                if se_geometry.contains(point):
                    nodes_in_se.append(node_id)
        
        if not nodes_in_se:
            print("⚠ No nodes found in Sé parish, using full city")
            return
        
        # Create subgraph
        self.graph = self.graph.subgraph(nodes_in_se).copy()
        print(f"✓ Filtered to {len(self.graph.nodes())} nodes in {self.target_parish} parish")
        
        # Filter POIs
        graph_nodes = set(self.graph.nodes())
        filtered_pois = {}
        
        for category, poi_list in self.pois.items():
            filtered_poi_list = []
            for poi_data in poi_list:
                if isinstance(poi_data, tuple):
                    node_id, poi_type = poi_data
                    if node_id in graph_nodes:
                        filtered_poi_list.append(poi_data)
                elif poi_data in graph_nodes:
                    filtered_poi_list.append(poi_data)
            filtered_pois[category] = filtered_poi_list
        
        self.pois = filtered_pois
        total_filtered_pois = sum(len(poi_list) for poi_list in self.pois.values())
        print(f"✓ Filtered to {total_filtered_pois} POIs in {self.target_parish} parish")
    
    def create_llm_decision_prompt(self, resident_info, available_pois, current_time):
        """Create a prompt for LLM to decide which POI to visit."""
        prompt = f"""You are a resident of Macau living in Sé parish. Based on your profile and current situation, decide which Point of Interest (POI) you want to visit next.

RESIDENT PROFILE:
- Age: {resident_info.get('age', 'Unknown')}
- Gender: {resident_info.get('gender', 'Unknown')}
- Persona: {resident_info.get('persona', 'Unknown')}
- Current mood: {resident_info.get('mood', 'neutral')}
- Current needs: {resident_info.get('needs', [])}
- Current time: {current_time}

AVAILABLE POIs IN SÉ PARISH:
"""
        
        # Add available POIs by category
        for category, poi_list in available_pois.items():
            if poi_list:
                prompt += f"\n{category.upper()}:\n"
                for i, (node_id, poi_type) in enumerate(poi_list[:5]):  # Limit to 5 per category
                    prompt += f"  - {poi_type} (ID: {node_id})\n"
        
        prompt += """
DECISION TASK:
Choose ONE POI to visit based on your current needs and preferences. Consider:
1. Your current needs (hunger, health, entertainment, etc.)
2. The time of day
3. Your persona characteristics
4. Distance and accessibility

RESPONSE FORMAT:
Provide your decision as a JSON object with:
{
    "chosen_poi": "poi_type",
    "poi_id": node_id,
    "reason": "brief explanation of why you chose this POI"
}

Example:
{
    "chosen_poi": "restaurant",
    "poi_id": 12345,
    "reason": "I'm feeling hungry and it's lunch time"
}

Your decision:"""
        
        return prompt
    
    def get_llm_poi_decision(self, resident_info, available_pois, current_time):
        """Get LLM decision for POI selection."""
        if not self.llm_wrapper:
            # Mock decision if LLM is not available
            return self.get_mock_poi_decision(available_pois)
        
        try:
            prompt = self.create_llm_decision_prompt(resident_info, available_pois, current_time)
            response = self.llm_wrapper.get_response(prompt)
            
            # Try to parse JSON response
            try:
                import json
                # Extract JSON from response if it's embedded in text
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    decision = json.loads(json_str)
                    
                    # Validate decision format
                    if 'chosen_poi' in decision and 'poi_id' in decision:
                        return decision
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠ Could not parse LLM response as JSON: {e}")
                print(f"  Raw response: {response[:200]}...")
                return self.get_mock_poi_decision(available_pois)
                
        except Exception as e:
            print(f"⚠ Error getting LLM decision: {e}")
            return self.get_mock_poi_decision(available_pois)
    
    def get_mock_poi_decision(self, available_pois):
        """Generate a mock POI decision when LLM is not available."""
        # Flatten all POIs
        all_pois = []
        for category, poi_list in available_pois.items():
            for node_id, poi_type in poi_list:
                all_pois.append((node_id, poi_type, category))
        
        if not all_pois:
            return None
        
        # Random selection
        chosen = random.choice(all_pois)
        return {
            "chosen_poi": chosen[1],
            "poi_id": chosen[0],
            "reason": f"Mock decision: randomly selected {chosen[1]} from {chosen[2]} category"
        }
    
    def setup_data_logging(self):
        """Set up comprehensive data logging."""
        print("Setting up data logging...")
        
        # Create logging configuration
        config = DataManagementConfig.create_comprehensive_config()
        self.simulation_id = f"se_parish_zhipuai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config.simulation_id = self.simulation_id
        config.output_directory = "output/se_parish_test"
        
        # Initialize data toolkit with both required parameters
        self.data_toolkit = DataManagementLoggingToolkit(
            output_dir=config.output_directory,
            simulation_id=self.simulation_id
        )
        print("✓ Data logging configured")
    
    def create_test_residents(self):
        """Create 10 test residents with diverse personas."""
        print("Creating 10 test residents...")
        
        # Define diverse personas for the residents
        personas = [
            PersonaType.ELDERLY_RESIDENT,
            PersonaType.WORKING_PARENT,
            PersonaType.YOUNG_PROFESSIONAL,
            PersonaType.STUDENT,
            PersonaType.COMMUNITY_LEADER,
            PersonaType.COMMUNITY_LEADER,
            PersonaType.POLICY_MAKER,
            PersonaType.HEALTHCARE_WORKER,
            PersonaType.WORKING_PARENT,  # Second working parent
            PersonaType.YOUNG_PROFESSIONAL  # Second young professional
        ]
        
        residents = []
        graph_nodes = list(self.graph.nodes())
        
        for i in range(self.num_residents):
            # Random starting location in Sé parish
            start_node = random.choice(graph_nodes)
            
            # Create resident with specific persona
            resident_info = {
                'id': f'se_resident_{i+1}',
                'age': random.randint(18, 80),
                'gender': random.choice(['male', 'female']),
                'persona': personas[i],
                'start_location': start_node,
                'mood': random.choice(['happy', 'neutral', 'stressed', 'excited']),
                'needs': random.sample(['hunger', 'health', 'entertainment', 'social', 'shopping'], 
                                     random.randint(1, 3))
            }
            
            residents.append(resident_info)
            print(f"  - Resident {i+1}: {resident_info['persona'].value}, age {resident_info['age']}, mood: {resident_info['mood']}")
        
        return residents
    
    def run_simulation(self):
        """Run the main simulation."""
        print(f"\nStarting simulation with {self.num_residents} residents in {self.target_parish} parish...")
        self.results["start_time"] = datetime.now()
        
        # Create residents
        residents = self.create_test_residents()
        self.results["residents"] = residents
        
        # Simulate decision-making for each time step
        for step in range(self.simulation_steps):
            current_time = f"{8 + step // 60:02d}:{step % 60:02d}"  # Start at 8:00 AM
            
            if step % 5 == 0:  # Progress update every 5 steps
                print(f"  Step {step}/{self.simulation_steps} - Time: {current_time}")
            
            # Each resident makes a decision
            for resident in residents:
                decision = self.get_llm_poi_decision(resident, self.pois, current_time)
                
                if decision:
                    # Record the decision
                    decision_record = {
                        "step": step,
                        "time": current_time,
                        "resident_id": resident['id'],
                        "decision": decision,
                        "resident_state": {
                            "mood": resident['mood'],
                            "needs": resident['needs'],
                            "location": resident.get('current_location', resident['start_location'])
                        }
                    }
                    
                    self.results["llm_decisions"].append(decision_record)
                    
                    # Simulate POI visit
                    poi_visit = {
                        "step": step,
                        "time": current_time,
                        "resident_id": resident['id'],
                        "poi_type": decision['chosen_poi'],
                        "poi_id": decision['poi_id'],
                        "reason": decision['reason']
                    }
                    
                    self.results["poi_visits"].append(poi_visit)
                    
                    # Update resident location
                    resident['current_location'] = decision['poi_id']
                    
                    # Log to data toolkit if available
                    if self.data_toolkit:
                        # Create a serializable copy of the resident state for logging
                        serializable_resident_state = resident.copy()
                        if 'persona' in serializable_resident_state and hasattr(serializable_resident_state['persona'], 'value'):
                            serializable_resident_state['persona'] = serializable_resident_state['persona'].value

                        self.data_toolkit.llm_logger.log_llm_interaction(
                            self.simulation_id, step, 
                            resident['id'], {
                                'interaction_id': f"step_{step}_resident_{resident['id']}",
                                'model_name': self.zhipuai_model,
                                'prompt_template': 'POI selection decision',
                                'full_prompt': f"POI selection for {resident['id']}",
                                'response': str(decision),
                                'agent_state': serializable_resident_state,
                                'decision_context': {'available_pois': len([poi for poi_list in self.pois.values() for poi in poi_list])},
                                'policy_contexts': [],
                                'latency_ms': 0.0,
                                'input_tokens': 0,
                                'output_tokens': 0,
                                'total_tokens': 0,
                                'cost_estimate': 0.0
                            }
                        )
        
        self.results["end_time"] = datetime.now()
        print(f"✓ Simulation completed!")
    
    def analyze_results(self):
        """Analyze and display simulation results."""
        print("\n" + "="*60)
        print("SIMULATION RESULTS ANALYSIS")
        print("="*60)
        
        # Basic statistics
        duration = self.results["end_time"] - self.results["start_time"]
        total_decisions = len(self.results["llm_decisions"])
        total_visits = len(self.results["poi_visits"])
        
        print(f"Simulation Duration: {duration}")
        print(f"Total LLM Decisions: {total_decisions}")
        print(f"Total POI Visits: {total_visits}")
        print(f"Average Decisions per Resident: {total_decisions / self.num_residents:.1f}")
        
        # POI popularity analysis
        poi_counts = {}
        for visit in self.results["poi_visits"]:
            poi_type = visit["poi_type"]
            poi_counts[poi_type] = poi_counts.get(poi_type, 0) + 1
        
        print(f"\nMOST POPULAR POI TYPES:")
        for poi_type, count in sorted(poi_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {poi_type}: {count} visits")
        
        # Resident activity analysis
        resident_activity = {}
        for decision in self.results["llm_decisions"]:
            resident_id = decision["resident_id"]
            if resident_id not in resident_activity:
                resident_activity[resident_id] = []
            resident_activity[resident_id].append(decision)
        
        print(f"\nRESIDENT ACTIVITY SUMMARY:")
        for resident in self.results["residents"]:
            resident_id = resident["id"]
            decisions = resident_activity.get(resident_id, [])
            print(f"  {resident_id} ({resident['persona'].value}): {len(decisions)} decisions")
        
        # Time-based analysis
        hourly_activity = {}
        for decision in self.results["llm_decisions"]:
            hour = decision["time"].split(":")[0]
            hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
        
        print(f"\nHOURLY ACTIVITY DISTRIBUTION:")
        for hour in sorted(hourly_activity.keys()):
            count = hourly_activity[hour]
            bar = "█" * (count // 5) + "▌" * (1 if count % 5 >= 3 else 0)
            print(f"  {hour}:00 - {count:3d} decisions {bar}")
    
    def save_results(self):
        """Save results to files."""
        print("\nSaving results...")
        
        # Create output directory
        output_dir = "output/se_parish_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results as JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"{output_dir}/se_parish_simulation_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        json_results = self.results.copy()
        json_results["start_time"] = self.results["start_time"].isoformat()
        json_results["end_time"] = self.results["end_time"].isoformat()
        
        # Convert PersonaType enums to strings
        for resident in json_results["residents"]:
            resident["persona"] = resident["persona"].value
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to: {results_file}")
        
        # Save summary report
        summary_file = f"{output_dir}/summary_report_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("SÉ PARISH SIMULATION SUMMARY REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Parish: {self.target_parish}\n")
            f.write(f"Number of Residents: {self.num_residents}\n")
            f.write(f"Simulation Steps: {self.simulation_steps}\n")
            f.write(f"LLM Model: {self.zhipuai_model}\n")
            f.write(f"LLM Platform: {self.zhipuai_platform}\n\n")
            
            # Add analysis results
            f.write("ANALYSIS RESULTS:\n")
            f.write(f"Total LLM Decisions: {len(self.results['llm_decisions'])}\n")
            f.write(f"Total POI Visits: {len(self.results['poi_visits'])}\n")
            f.write(f"Average Decisions per Resident: {len(self.results['llm_decisions']) / self.num_residents:.1f}\n")
        
        print(f"✓ Summary report saved to: {summary_file}")
    
    def run_full_test(self):
        """Run the complete test suite."""
        print("STARTING SÉ PARISH ZHIPUAI TEST")
        print("="*50)
        
        try:
            # Step 1: Setup ZhipuAI LLM
            self.setup_zhipuai_llm()
            
            # Step 2: Load city data
            self.load_city_data()
            
            # Step 3: Filter to Sé parish
            self.filter_to_se_parish()
            
            # Step 4: Setup data logging
            self.setup_data_logging()
            
            # Step 5: Run simulation
            self.run_simulation()
            
            # Step 6: Analyze results
            self.analyze_results()
            
            # Step 7: Save results
            self.save_results()
            
            print("\n" + "="*50)
            print("TEST COMPLETED SUCCESSFULLY!")
            print("="*50)
            
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def main():
    """Main function to run the test."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create and run test
    test_runner = SeParishTestRunner()
    success = test_runner.run_full_test()
    
    if success:
        print("\nTest completed successfully! Check the output directory for detailed results.")
    else:
        print("\nTest failed. Please check the error messages above.")
    
    return success


if __name__ == "__main__":
    main() 