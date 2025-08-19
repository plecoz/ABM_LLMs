from agents.base_person_agent import BaseAgent
import random
import networkx as nx
import logging
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime
import copy
import re
from brains.concordia_brain import ConcordiaBrain
from agents.action_system import Action, get_available_actions, EVERYDAY_ACTIONS


class Resident(BaseAgent):
    """
    Resident agent representing a person living in the fifteen-minute city.
    
    This class is organized into the following modules:
    1. Initialization and Agent Properties
    2. Path Selection and Travel
    3. Movement and Target Selection  
    4. Needs Management
    5. Action System
    6. Memory and State Management
    7. Main Step Logic
    """
    
    # =====================================================================
    # 1. INITIALIZATION AND AGENT PROPERTIES MODULE
    # =====================================================================
    
    def __init__(self, model, unique_id, geometry, home_node, accessible_nodes, **kwargs):
        """
        Initialize a resident agent.
        
        Args:
            model: Model instance the agent belongs to
            unique_id: Unique identifier for the agent
            geometry: Shapely geometry object representing the agent's location
            home_node: The agent's home node in the network
            accessible_nodes: Dictionary of nodes accessible to the agent
            **kwargs: Additional agent properties that can be customized
        """
        # Pass parameters in the correct order to parent class
        super().__init__(model, unique_id, geometry, **kwargs)
        
        # Core location and mobility attributes (keep separate for frequent access)
        self.home_node = home_node
        self.current_node = home_node
        self.accessible_nodes = accessible_nodes
        self.visited_pois = []
        self.mobility_mode = "walk"
        self.last_visited_node = None
        
        # Household and occupation (keep separate as requested)
        self.household_members = kwargs.get('household_members', [])
        self.household_type = kwargs.get('household_type', "single")
        self.economic_status = kwargs.get('economic_status', "employed")
        
        # Consolidated attributes dictionary
        self.attributes = {
            # Demographics
            'age': kwargs.get('age', 30),
            'age_class': kwargs.get('age_class', None),
            'gender': kwargs.get('gender', 'male'),
            'income': kwargs.get('income', 50000),
            'education': kwargs.get('education', 'high_school'),
            'occupation': kwargs.get('occupation', None),
            'industry': kwargs.get('industry', None),
            
            # Location and social
            'parish': kwargs.get('parish', None),
            'family_id': kwargs.get('family_id', None),
            'social_network': kwargs.get('social_network', []),
            'daily_schedule': kwargs.get('daily_schedule', {}),
            'personality_traits': kwargs.get('personality_traits', {}),
            'activity_preferences': kwargs.get('activity_preferences', {}),
            
            # Physical attributes
            'access_distance': kwargs.get('access_distance', 0),
        }
        
        # Convenience properties for frequently accessed attributes
        self.age = self.attributes['age']
        self.parish = self.attributes['parish']
        
        
        # Determine step size based on age for calculating travel times
        is_elderly = False
        if self.attributes['age_class']:
            age_class_str = str(self.attributes['age_class']).lower()
            if any(s in age_class_str for s in ['65+', '65-', '70+', '70-', '75+', '75-', '80+', '80-', '85+', '85-']):
                is_elderly = True
        
        self.step_size = 60.0 if is_elderly else 80.0  # meters per minute

        # Calculate home access time penalty based on distance from building to network
        if self.attributes['access_distance'] > 0.1:  # Only apply penalty for distances > 0.1 meters
            # Time (in steps/minutes) to walk from building to nearest street node
            # We use ceil to ensure any non-zero distance results in at least a 1-minute penalty
            self.home_access_time = math.ceil(self.attributes['access_distance'] / self.step_size)
            # print(f"DEBUG: Resident {self.unique_id} has a home access time penalty of {self.home_access_time} minutes.")
        else:
            self.home_access_time = 0
        
        # Mobility constraints - speed in km/h
        if self.age >= 65:
            self.speed = 3.0  # Elderly walk at 3 km/h
        else:
            self.speed = 5.0  # Everyone else walks at 5 km/h
        
        # Travel time tracking
        self.traveling = False
        self.travel_time_remaining = 0
        self.destination_node = None
        self.destination_geometry = None
        
        # Simple action system
        self.performing_action = False
        self.current_action: Optional[Action] = None  # Current action being performed
        self.action_time_remaining = 0
        self.action_memory = []  # List of (action_name, timestamp) tuples - complete history
        self.pending_action: Optional[Action] = None  # Action to start after arriving at POI
        # Tracking for healthcare accessibility across runs
        self.healthcare_access_success = None
        # Track reasons for healthcare access failures
        self.failure_counts = {"distance": 0, "capacity": 0, "cost": 0}
        # Check employment status (case-insensitive)
        economic_status = kwargs.get('economic_status', 'unemployed')
        self.is_employed = str(economic_status).lower() == 'employed'
        print(f"Agent {unique_id}: age={self.attributes.get('age', 'unknown')}, economic_status='{economic_status}' -> is_employed={self.is_employed}")
        
        # Energy and money
        
        self.daily_income = kwargs.get('daily_income', 200.0 if self.is_employed else 50.0)  # Daily income
        self.money = self.daily_income  # Start with one day's income
        self.last_paid_day = -1  # Track last day we received income
        
        # Health status
        self.health_status = "sane"  # "sane", "sick", or "cured"
        
        # Path selection for LLM agents
        self.selected_travel_path = None  # Store the path selected by LLM
        self.path_selection_history = []  # Track path choices for learning

        # demographics attributes 
        self.demographics = None
        # Load parish-specific demographics if provided
        self.parish_demographics = kwargs.get('parish_demographics', {})
        
        # Enhanced memory module
        self.memory = {
            'income': self.attributes['income'],
            'visited_pois': [],  # List of dicts: {step, poi_id, poi_type, category, income}
            'interactions': [],  # List of interaction records
            'historical_needs': [],  # List of needs over time: {step, needs_dict}
            'completed_actions': [],  # List of completed actions with outcomes
        }
        
        # Calculate daily income from monthly income (moved here to ensure it uses final income value)
        self.daily_income = self.attributes['income'] / 30
        
        
        # Initialize logger if not provided
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"Resident-{unique_id}")

        # ---------------------------------------------------
        # Concordia Brain (LLM) integration
        # ---------------------------------------------------
        # Initialize brain (Concordia integration)
        try:
            from brains.concordia_brain import ConcordiaBrain
            self.brain = ConcordiaBrain(name=f"Resident-{unique_id}")
        except Exception as e:

            self.brain = None

        # Initialize last path calculation time to prevent rapid recalculation
        self.last_path_calculation_time = 0
        self.path_calculation_cooldown = 5  # Minimum 5 steps between path calculations
        
        # Path selection tracking for analysis
        self.path_selection_stats = {
            'total_multi_path_decisions': 0,  # Total times multiple paths were available
            'shortest_path_not_selected': 0,  # Times shortest path was not chosen
            'concordia_decisions': 0,  # Times Concordia brain made the decision
            'fallback_decisions': 0   # Times fallback was used
        }

    # =====================================================================
    # AGENT PROPERTY GENERATION METHODS
    # =====================================================================

    @staticmethod
    def _generate_agent_properties(parish=None, demographics=None, parish_demographics=None, 
                                 city=None, industry_distribution=None, occupation_distribution=None, 
                                 income_distribution=None, economic_status_distribution=None):
        """
        Generate agent properties based on demographic distributions.
        For Macau, follows strict logic sequence:
        1. Use parish_demographic.json for age, gender, education_level
        2. Use economic_status.json based on age group to assign economic_status
        3. If economic_status is "Employed", use occupation.json based on age group
        4. Use industry.json based on education to assign industry
        5. If occupation assigned, use income.json; otherwise random low income
        
        Args:
            parish: The parish name to use for demographics (optional)
            demographics: Main demographics data
            parish_demographics: Parish-specific demographics data
            city: City name for location-specific logic
            industry_distribution: Industry distribution data
            occupation_distribution: Occupation distribution data
            income_distribution: Income distribution data
            economic_status_distribution: Economic status distribution data
            
        Returns:
            Dictionary of agent properties
        """

        if parish and parish_demographics and parish in parish_demographics:
            demographics = parish_demographics[parish]
            # print(f"DEBUG: Using parish-specific demographics for {parish}")
        
        if not demographics:
            # print("DEBUG: No demographics available, returning empty dict")
            return {}
        
        props = {}

        # === STEP 1: Use parish_demographic.json for age, gender, education ===

        # --- AGE & AGE CLASS ---
        age_dist = demographics.get('age_distribution', {})
        age_group = Resident._sample_from_distribution(age_dist)
        props['age_class'] = age_group  # Store the sampled age class for reference
        # print(f"DEBUG: Step 1 - Sampled age_class: {age_group}")

        # Convert age group to an actual age value
        if age_group is not None:
            if '-' in age_group:  # e.g. "25-29"
                min_age, max_age = age_group.split('-')
                try:
                    min_age = int(min_age)
                    max_age = int(max_age)
                    props['age'] = random.randint(min_age, max_age)
                except ValueError:
                    props['age'] = random.randint(18, 90)
            elif age_group.endswith('+'):  # e.g. "85+"
                try:
                    min_age = int(age_group.rstrip('+'))
                    props['age'] = random.randint(min_age, min_age + 10)
                except ValueError:
                    props['age'] = random.randint(65, 90)
            else:  # Unknown pattern – fallback
                props['age'] = random.randint(18, 90)
        else:
            props['age'] = random.randint(18, 90)
        
        # print(f"DEBUG: Step 1 - Generated age: {props['age']}")

        # --- GENDER ---
        gender_dist = {}
        if demographics.get('gender_by_age') and age_group in demographics['gender_by_age']:
            gender_dist = demographics['gender_by_age'][age_group]
        else:
            gender_dist = demographics.get('gender_distribution', {})
        props['gender'] = Resident._sample_from_distribution(gender_dist)
        # print(f"DEBUG: Step 1 - Generated gender: {props['gender']}")

        # --- EDUCATION LEVEL ---
        education_level = None
        if (demographics.get('education_distribution_by_age_and_gender') and
            age_group in demographics['education_distribution_by_age_and_gender'] and
            props['gender'] in demographics['education_distribution_by_age_and_gender'][age_group]):
            edu_dist_raw = demographics['education_distribution_by_age_and_gender'][age_group][props['gender']]
            # Flatten nested structures (e.g. Primary education -> complete/incomplete)
            flattened_edu_dist = {}
            for k, v in edu_dist_raw.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flattened_edu_dist[f"{k} {sub_k}"] = sub_v
                else:
                    flattened_edu_dist[k] = v
            education_level = Resident._sample_from_distribution(flattened_edu_dist)
        else:
            education_dist = demographics.get('education_distribution', {})
            education_level = Resident._sample_from_distribution(education_dist)
        props['education'] = education_level
        # print(f"DEBUG: Step 1 - Generated education: '{education_level}'")

        # For non-Macau cities, use simplified logic
        if city != "Macau, China":
            # Default values for non-Macau cities
            props['employment_status'] = "employed"
            props['household_type'] = "single"
            props['industry'] = None
            props['occupation'] = None
            props['income'] = random.randint(30000, 100000)  # Default medium income
            # print(f"DEBUG: Non-Macau city, using simplified logic")
            # print(f"DEBUG: Final generated properties: {props}")
            # print("---")
            return props

        # === MACAU-SPECIFIC LOGIC ===
        # print(f"DEBUG: Starting Macau-specific logic")

        # === STEP 2: Use economic_status.json based on age group ===
        economic_status = None
        # Load economic_status distribution from parameter
        if economic_status_distribution:
            # Map age to age group used in economic_status.json
            economic_age_group = None
            age = props['age']
            
            # Children under 16 are not employed
            if age < 18:
                economic_status = "Student"  # Children are students, not employed
                economic_age_group = None  # Skip the rest of the logic
            elif 18 <= age <= 24:
                economic_age_group = "16-24"
            elif 25 <= age <= 29:
                economic_age_group = "25-29"
            elif 30 <= age <= 34:
                economic_age_group = "30-34"
            elif 35 <= age <= 39:
                economic_age_group = "35-39"
            elif 40 <= age <= 44:
                economic_age_group = "40-44"
            elif 45 <= age <= 49:
                economic_age_group = "45-49"
            elif 50 <= age <= 54:
                economic_age_group = "50-54"
            elif 55 <= age <= 59:
                economic_age_group = "55-59"
            elif 60 <= age <= 64:
                economic_age_group = "60-64"
            elif age >= 65:
                economic_age_group = ">=65"
            
            # Map gender to economic_status.json format (M/F)
            gender_key = "M" if props['gender'].lower() == 'male' else "F"
            
            # print(f"DEBUG: Step 2 - Age: {age}, mapped to economic_age_group: '{economic_age_group}'")
            # print(f"DEBUG: Step 2 - Gender: {props['gender']}, mapped to gender_key: '{gender_key}'")
            
            # Get economic status distribution for this age group and gender
            if (economic_age_group and economic_age_group in economic_status_distribution and 
                gender_key in economic_status_distribution[economic_age_group]):
                
                econ_dist = economic_status_distribution[economic_age_group][gender_key]
                economic_status = Resident._sample_from_distribution(econ_dist)
                # print(f"DEBUG: Step 2 - Selected economic_status: '{economic_status}'")
            else:
                # print(f"DEBUG: Step 2 - No economic status distribution found for age_group: '{economic_age_group}', gender: '{gender_key}'")
                economic_status = "Employed"  # Default fallback
        else:
            # print(f"DEBUG: Step 2 - No economic_status_distribution available, using education-based fallback")
            # Fallback: determine employment status based on education level
            if education_level:
                edu_lower = education_level.lower()
                if 'tertiary' in edu_lower:
                    economic_status = "Employed" if random.random() < 0.8 else "Unemployed"
                elif 'secondary' in edu_lower and 'senior' in edu_lower:
                    economic_status = "Employed" if random.random() < 0.6 else "Unemployed"
                elif 'diploma' in edu_lower:
                    economic_status = "Employed" if random.random() < 0.75 else "Unemployed"
                else:
                    economic_status = "Employed" if random.random() < 0.4 else "Unemployed"
            else:
                economic_status = "Employed" if random.random() < 0.5 else "Unemployed"
        
        props['economic_status'] = economic_status  # Add this so it gets passed to constructor

        props['employment_status'] = economic_status.lower() if economic_status else "unemployed"
        
        # print(f"DEBUG: Step 2 - Generated economic_status: '{economic_status}'")

        # === STEP 3: If economic_status is "Employed", assign occupation based on age group ===
        occupation_choice = None
        if economic_status == "Employed" and occupation_distribution:
            age = props['age']
            # print(f"DEBUG: Step 3 - Attempting to assign occupation for employed person, age: {age}")
            
            # Map age to age group used in occupation.json
            occupation_age_group = None
            if 16 <= age <= 24:
                occupation_age_group = "16-24"
            elif 25 <= age <= 29:
                occupation_age_group = "25-29"
            elif 30 <= age <= 34:
                occupation_age_group = "30-34"
            elif 35 <= age <= 39:
                occupation_age_group = "35-39"
            elif 40 <= age <= 44:
                occupation_age_group = "40-44"
            elif 45 <= age <= 49:
                occupation_age_group = "45-49"
            elif 50 <= age <= 54:
                occupation_age_group = "50-54"
            elif 55 <= age <= 59:
                occupation_age_group = "55-59"
            elif 60 <= age <= 64:
                occupation_age_group = "60-64"
            elif age >= 65:
                occupation_age_group = "≧65"
            
            # print(f"DEBUG: Step 3 - Mapped age {age} to occupation_age_group: '{occupation_age_group}'")
            # print(f"DEBUG: Step 3 - Available occupation age groups: {list(occupation_distribution.keys())}")
            
            if occupation_age_group and occupation_age_group in occupation_distribution:
                occupation_dist = occupation_distribution[occupation_age_group]
                occupation_choice = Resident._sample_from_distribution(occupation_dist)
                # print(f"DEBUG: Step 3 - Selected occupation: '{occupation_choice}'")
            else:
                # print(f"DEBUG: Step 3 - No occupation distribution found for age_group: '{occupation_age_group}'")
                pass
        else:
            # print(f"DEBUG: Step 3 - Skipping occupation assignment (economic_status: '{economic_status}')")
            pass
        
        props['occupation'] = occupation_choice

        # === STEP 4: Assign industry based on education ===
        industry_choice = None
        if education_level and industry_distribution:
            # print(f"DEBUG: Step 4 - Attempting to assign industry for education: '{education_level}'")
            norm_edu = Resident._normalise_string(education_level)
            # print(f"DEBUG: Step 4 - Normalized education: '{norm_edu}'")
            # print(f"DEBUG: Step 4 - Available industry keys: {list(industry_distribution.keys())}")
            
            # Direct match first
            industry_dist = industry_distribution.get(norm_edu)
            if not industry_dist:
                # print(f"DEBUG: Step 4 - No direct match found, trying partial matching...")
                # Try partial matching
                for key, dist in industry_distribution.items():
                    normalized_key = Resident._normalise_string(key)
                    # print(f"DEBUG: Step 4 - Comparing '{norm_edu}' with '{normalized_key}'")
                    if normalized_key == norm_edu:
                        industry_dist = dist
                        # print(f"DEBUG: Step 4 - Found partial match with key: '{key}'")
                        break
            else:
                # print(f"DEBUG: Step 4 - Found direct match for normalized education")
                pass
                
            if industry_dist:
                industry_choice = Resident._sample_from_distribution(industry_dist)
                # print(f"DEBUG: Step 4 - Selected industry: '{industry_choice}'")
            else:
                # print(f"DEBUG: Step 4 - No industry distribution found for education: '{education_level}'")
                pass
        else:
            # print(f"DEBUG: Step 4 - Skipping industry assignment (education: '{education_level}', industry_distribution available: {industry_distribution is not None})")
            pass
        
        props['industry'] = industry_choice

        # === STEP 5: Assign income based on occupation or random low if no occupation ===
        if occupation_choice and income_distribution:
            # print(f"DEBUG: Step 5 - Attempting to assign income based on occupation: '{occupation_choice}'")
            # print(f"DEBUG: Step 5 - Available income keys: {list(income_distribution.keys())}")
            
            if occupation_choice in income_distribution:
                income_bracket_dist = income_distribution[occupation_choice]
                income_bracket = Resident._sample_from_distribution(income_bracket_dist)
                # print(f"DEBUG: Step 5 - Selected income bracket: '{income_bracket}'")
                
                # Convert income bracket to actual income value
                if income_bracket:
                    actual_income = Resident._convert_income_bracket_to_value(income_bracket)
                    # print(f"DEBUG: Step 5 - Converted income bracket '{income_bracket}' to actual income: {actual_income}")
                    if actual_income is not None:
                        props['income'] = actual_income
                        # print(f"DEBUG: Step 5 - Final income assigned: {props['income']}")
                    else:
                        # print(f"DEBUG: Step 5 - Failed to convert income bracket, using random low income")
                        props['income'] = random.randint(10000, 30000)  # Low income fallback
                else:
                    # print(f"DEBUG: Step 5 - No income bracket selected, using random low income")
                    props['income'] = random.randint(10000, 30000)  # Low income fallback
            else:
                # print(f"DEBUG: Step 5 - Occupation '{occupation_choice}' not found in income distribution, using random low income")
                props['income'] = random.randint(10000, 30000)  # Low income fallback
        else:
            # print(f"DEBUG: Step 5 - No occupation assigned, using random low income")
            props['income'] = random.randint(10000, 30000)  # Low income for unemployed/no occupation

        # Default values for additional attributes
        props['household_type'] = "single"
        
        # print(f"DEBUG: Final generated properties: {props}")
        # print("---")

        return props

    @staticmethod
    def _sample_from_distribution(distribution):
        """
        Sample a value from a probability distribution.
        
        Args:
            distribution: Dictionary mapping values to probabilities
            
        Returns:
            A sampled value from the distribution
        """
        if not distribution:
            return None
            
        items = list(distribution.keys())
        probabilities = list(distribution.values())
        
        # Normalize probabilities if they don't sum to 1
        prob_sum = sum(probabilities)
        if prob_sum != 1.0 and prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]
        
        return random.choices(items, probabilities)[0]

    @staticmethod
    def _normalise_string(s):
        """Normalize a string by removing spaces, dashes, and converting to lowercase."""
        if not s:
            return ""
        # Remove spaces, dashes, and other common punctuation
        normalized = re.sub(r'[\s\-_.,;:()]+', '', s.lower())
        return normalized

    @staticmethod
    def _convert_income_bracket_to_value(income_bracket):
        """Convert income bracket string to actual income value."""
        if not income_bracket:
            return None
        
        # Handle Macau income bracket formats (uses spaces as thousands separators)
        bracket = income_bracket.strip()
        
        # Handle "< X XXX" format
        if bracket.startswith("< "):
            max_val = int(bracket[2:].replace(" ", ""))
            return random.randint(1000, max_val)
        
        # Handle "≧X XXX" format (greater than or equal to)
        elif bracket.startswith("≧"):
            min_val = int(bracket[1:].replace(" ", ""))
            return random.randint(min_val, min_val + 50000)  # Add reasonable upper bound
        
        # Handle "X XXX - Y YYY" format
        elif " - " in bracket:
            parts = bracket.split(" - ")
            if len(parts) == 2:
                min_val = int(parts[0].replace(" ", ""))
                max_val = int(parts[1].replace(" ", ""))
                return random.randint(min_val, max_val)
        
        # Handle "Unpaid family worker" or other special cases
        elif "unpaid" in bracket.lower():
            return 0
        
        # Legacy handling for comma-separated formats (keep for backward compatibility)
        elif "," in bracket:
            if "10,000" in bracket and "19,999" in bracket:
                return random.randint(10000, 19999)
            elif "20,000" in bracket and "29,999" in bracket:
                return random.randint(20000, 29999)
            elif "30,000" in bracket and "39,999" in bracket:
                return random.randint(30000, 39999)
            elif "40,000" in bracket and "49,999" in bracket:
                return random.randint(40000, 49999)
            elif "50,000" in bracket and "59,999" in bracket:
                return random.randint(50000, 59999)
            elif "60,000" in bracket and "69,999" in bracket:
                return random.randint(60000, 69999)
            elif "70,000" in bracket and "79,999" in bracket:
                return random.randint(70000, 79999)
            elif "80,000" in bracket and "89,999" in bracket:
                return random.randint(80000, 89999)
            elif "90,000" in bracket and "99,999" in bracket:
                return random.randint(90000, 99999)
            elif "100,000" in bracket:
                return random.randint(100000, 200000)
        
        # Default fallback
        return 50000

        # ---------------------------------------------------
        # Concordia Brain (LLM) integration
        # ---------------------------------------------------

    # =====================================================================
    # 2. PATH SELECTION AND TRAVEL MODULE
    # =====================================================================

    def _calculate_standard_travel_time(self, from_node, to_node):
        """
        Calculate travel time using standard shortest path (for non-LLM agents).
        
        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            
        Returns:
            Number of time steps needed for travel
        """
        # Always calculate the actual shortest path length for consistency
        try:
            # Calculate shortest path length along the street network
            distance_meters = nx.shortest_path_length(
                self.model.graph, 
                from_node, 
                to_node, 
                weight='length'
            )
        except (nx.NetworkXNoPath, KeyError):
            self.logger.warning(f"No path found from {from_node} to {to_node}")
            return None

        # Determine step size based on age_class
        # Check if age_class indicates elderly (65+ years)
        is_elderly = False
        if self.attributes['age_class']:
            age_class_str = str(self.attributes['age_class']).lower()
            # Check for age classes that indicate 65+ years
            if ('65+' in age_class_str or '65-' in age_class_str or 
                '70+' in age_class_str or '70-' in age_class_str or
                '75+' in age_class_str or '75-' in age_class_str or
                '80+' in age_class_str or '80-' in age_class_str or
                '85+' in age_class_str or '85-' in age_class_str):
                is_elderly = True
        
        if is_elderly:
            step_size = 60.0  # 60 meters per minute for elderly
        else:
            step_size = 80.0  # 80 meters per minute for younger residents
        
        # Calculate number of steps needed
        # Always round UP to ensure no step exceeds the agent's step size
        steps_needed = math.ceil(distance_meters / step_size)
        
        # Ensure at least 1 time step
        return max(1, steps_needed)

    def _calculate_travel_time_with_path_selection(self, from_node, to_node):
        """
        Calculate travel time using LLM-based path selection from multiple alternatives.
        
        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            
        Returns:
            Number of time steps needed for travel using selected path
        """
        # Check cooldown to prevent rapid recalculation
        current_time = getattr(self.model, 'step_count', 0)
        if current_time - self.last_path_calculation_time < self.path_calculation_cooldown:
            return self._calculate_standard_travel_time(from_node, to_node)
        
        # Update last calculation time
        self.last_path_calculation_time = current_time
        
        # Circuit breaker to prevent infinite loops
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Get multiple path options
                path_options = self._get_multiple_path_options(from_node, to_node, max_paths=3)
                
                if not path_options:
                    self.logger.warning(f"No path options found from {from_node} to {to_node}")
                    return None
                
                # If only one path available, use it directly
                if len(path_options) == 1:
                    selected_path = path_options[0]
                else:
                    # Use LLM to score and select the best path
                    selected_path = self._select_path_with_llm(path_options, from_node, to_node)
                
                # Validate selected path
                if selected_path is None or not isinstance(selected_path, list) or len(selected_path) < 2:
                    retry_count += 1
                    continue
                
                # Store the selected path for actual travel
                self.selected_travel_path = selected_path
                
                # Calculate travel time based on selected path
                travel_time = self._calculate_time_for_path_nodes(selected_path)
                
                # Validate travel time
                if travel_time is None or travel_time <= 0:
                    retry_count += 1
                    continue
                
                return travel_time
                
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Error in LLM path selection (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    break
        
        # If all retries failed, fall back to standard calculation
        return self._calculate_standard_travel_time(from_node, to_node)

    def _get_multiple_path_options(self, from_node, to_node, max_paths=4):
        """
        Generate 4 shortest path options between two nodes.
        
        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            max_paths: Maximum number of paths to generate (default: 4)
        
        Returns:
            List of path dictionaries with OSM metadata for LLM scoring
        """
        try:
            # Generate k-shortest paths using NetworkX
            paths = self._get_k_shortest_paths(from_node, to_node, max_paths)
            
            # Extract OSM metadata for each path
            path_options = []
            
            for i, path_nodes in enumerate(paths):
                if path_nodes:  # Ensure path exists
                    path_data = self._extract_path_metadata(path_nodes, i + 1)
                    if path_data:
                        path_options.append(path_data)
            return path_options
            
        except Exception as e:
            self.logger.error(f"Error generating path options: {e}")
            return []

    def _get_k_shortest_paths(self, from_node, to_node, k):
        """
        Get k shortest paths using simple approach.
        
        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            k: Number of paths to find
            
        Returns:
            List of path node lists
        """
        try:
            paths = []
            graph = self.model.graph.copy()
            
            # Get the shortest path first
            try:
                shortest_path = nx.shortest_path(graph, from_node, to_node, weight='length')
                paths.append(shortest_path)
            except nx.NetworkXNoPath:
                return []
            
            # Try to find alternative paths by temporarily removing edges
            for attempt in range(k - 1):
                if len(paths) >= k:
                    break
                
                # Create a copy of the graph and remove some edges from previous paths
                temp_graph = graph.copy()
                
                # Remove some edges from existing paths to force alternatives
                for existing_path in paths:
                    # Ensure existing_path is actually a list of nodes
                    if not isinstance(existing_path, (list, tuple)) or len(existing_path) <= 2:
                        continue
                    
                    # Remove middle edges to force different routes
                    edges_to_remove = []
                    try:
                        for i in range(1, min(3, len(existing_path) - 1)):  # Remove 1-2 middle edges
                            if i < len(existing_path) - 1:
                                node1, node2 = existing_path[i], existing_path[i + 1]
                                # Ensure nodes are valid
                                if node1 is not None and node2 is not None:
                                    edges_to_remove.append((node1, node2))
                        
                        for edge in edges_to_remove:
                            if temp_graph.has_edge(edge[0], edge[1]):
                                temp_graph.remove_edge(edge[0], edge[1])
                    except (IndexError, TypeError) as e:
                        self.logger.warning(f"Error processing path for edge removal: {e}")
                        continue
                
                # Try to find a path in the modified graph
                try:
                    alt_path = nx.shortest_path(temp_graph, from_node, to_node, weight='length')
                    # Check if this path is sufficiently different
                    if not any(self._paths_are_same(alt_path, existing) for existing in paths):
                        paths.append(alt_path)
                except nx.NetworkXNoPath:
                    continue
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error in k-shortest paths: {e}")
            return []

    def _extract_path_metadata(self, path_nodes, path_number):
        """
        Extract OSM metadata from a path for LLM scoring.
        
        Args:
            path_nodes: List of node IDs in the path
            path_number: Path identifier number
            
        Returns:
            Dictionary with path metadata
        """
        try:
            graph = self.model.graph
            
            # Calculate basic metrics
            path_length = 0
            road_types = []
            surface_types = []
            max_speeds = []
            slopes = []
            speed_limits = []
            
            # Analyze each edge in the path
            for i in range(len(path_nodes) - 1):
                node1, node2 = path_nodes[i], path_nodes[i + 1]
                
                # Get edge data - handle both single edge and multi-edge cases
                edge_data = graph.get_edge_data(node1, node2)
                if edge_data is None:
                    continue
                
                # Handle MultiGraph case where edge_data might be a dict of dicts
                if isinstance(edge_data, dict) and not any(key in edge_data for key in ['length', 'highway']):
                    # This is likely a MultiGraph with multiple edges, take the first one
                    edge_data = list(edge_data.values())[0] if edge_data else {}
                
                # Extract length
                edge_length = edge_data.get('length', 0)
                path_length += edge_length
                
                # Extract road type (highway tag) - handle complex data types
                highway_type = edge_data.get('highway', 'unclassified')
                
                if isinstance(highway_type, (list, tuple)):
                    highway_type = highway_type[0] if highway_type else 'unclassified'
                elif isinstance(highway_type, dict):
                    highway_type = 'complex_type'  # Handle complex highway data
                elif not isinstance(highway_type, str):
                    highway_type = str(highway_type)
                
                road_types.append(str(highway_type))
                
                # Extract surface type if available - handle complex data types
                surface = edge_data.get('surface', 'unknown')
                
                if isinstance(surface, (list, tuple)):
                    surface = surface[0] if surface else 'unknown'
                elif isinstance(surface, dict):
                    surface = 'complex_surface'  # Handle complex surface data
                elif not isinstance(surface, str):
                    surface = str(surface)
                
                surface_types.append(str(surface))
                
                # Extract max speed if available - handle complex data types
                max_speed = edge_data.get('maxspeed', 'unknown')
                
                if isinstance(max_speed, (list, tuple)):
                    max_speed = max_speed[0] if max_speed else 'unknown'
                elif isinstance(max_speed, dict):
                    max_speed = 'complex_speed'  # Handle complex speed data
                elif not isinstance(max_speed, str):
                    max_speed = str(max_speed)
                
                max_speeds.append(str(max_speed))
                
                # Extract slope and speed limit using helper functions
                from environment.fifteenminutescity.city_network import get_slope_from_edge, get_speed_limit_from_edge
                
                slope = get_slope_from_edge(edge_data)
                if slope is not None:
                    slopes.append(slope)
                
                speed_limit = get_speed_limit_from_edge(edge_data)
                if speed_limit is not None:
                    speed_limits.append(speed_limit)
            
            # Calculate travel time
            travel_time_minutes = self._calculate_time_for_path_nodes(path_nodes)
            
            # Determine dominant road types
            road_type_counts = {}
            for road_type in road_types:
                if not isinstance(road_type, str):
                    road_type = str(road_type)
                road_type_counts[road_type] = road_type_counts.get(road_type, 0) + 1
            
            dominant_road_type = max(road_type_counts, key=road_type_counts.get) if road_type_counts else 'unknown'
            
            # Process slope and speed limit data
            avg_slope = sum(slopes) / len(slopes) if slopes else None
            max_speed_limit = max(speed_limits) if speed_limits else None
            
            # Create simplified metadata for LLM - safely handle sets with proper string conversion
            try:
                unique_road_types = list(set(road_types))
            except TypeError:
                # Fallback if any road_types are unhashable
                unique_road_types = list(dict.fromkeys(road_types))  # Remove duplicates preserving order
            
            try:
                unique_surface_types = list(set([s for s in surface_types if s != 'unknown']))
            except TypeError:
                # Fallback if any surface_types are unhashable
                filtered_surfaces = [s for s in surface_types if s != 'unknown']
                unique_surface_types = list(dict.fromkeys(filtered_surfaces))
            
            return {
                'path_id': path_number,
                'path_nodes': path_nodes,
                'distance_meters': round(path_length, 0),
                'travel_time_minutes': round(travel_time_minutes, 1),
                'dominant_road_type': dominant_road_type,
                'road_types': unique_road_types,
                'surface_types': unique_surface_types,
                'has_speed_limits': any(speed != 'unknown' for speed in max_speeds),
                'total_segments': len(path_nodes) - 1,
                'avg_slope': round(avg_slope, 1) if avg_slope is not None else None,
                'max_speed_limit': max_speed_limit
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting path metadata: {e}")
            return None


    def _calculate_time_for_path_nodes(self, path_nodes):
        """
        Calculate travel time for a list of path nodes.
        
        Args:
            path_nodes: List of node IDs in the path
            
        Returns:
            Travel time in minutes
        """
        total_distance = 0
        
        for i in range(len(path_nodes) - 1):
            # Handle both simple graphs and MultiGraphs properly
            edge_data = self.model.graph.get_edge_data(path_nodes[i], path_nodes[i + 1])
            
            if edge_data is None:
                # No edge found, use default distance
                edge_length = 100  # Default 100m if no data
            elif isinstance(edge_data, dict):
                # Check if this is a MultiGraph (dict of dicts) or simple graph (single dict)
                if any(key in edge_data for key in ['length', 'highway', 'osmid']):
                    # Simple graph - edge_data is the actual edge attributes
                    edge_length = edge_data.get('length', 100)
                else:
                    # MultiGraph - edge_data is a dict of edge keys, take the first one
                    first_edge_key = list(edge_data.keys())[0] if edge_data else 0
                    actual_edge_data = edge_data.get(first_edge_key, {})
                    edge_length = actual_edge_data.get('length', 100)
            else:
                # Unexpected data type
                edge_length = 100
            
            total_distance += edge_length
        
        # Use agent's step size to calculate time
        is_elderly = self.age >= 65
        step_size = 60.0 if is_elderly else 80.0
        # print(f"DEBUG: path time calculated!")
        
        return max(1, math.ceil(total_distance / step_size))


    def start_travel(self, target_node, target_geometry):
        """
        Start traveling to a target node.
        
        Args:
            target_node: Destination node ID
            target_geometry: Destination geometry
            
        Returns:
            Boolean indicating if travel was successfully started
        """
        travel_time = self._calculate_travel_time_with_path_selection(self.current_node, target_node)
        
        if travel_time is None:
            return False
        
        # Add access time penalty if starting from or going to home
        is_starting_from_home = self.current_node == self.home_node and not self.traveling
        is_going_home = target_node == self.home_node
        
        if (is_starting_from_home or is_going_home) and self.current_node != target_node:
            travel_time += self.home_access_time
            # print(f"DEBUG: Resident {self.unique_id} trip to/from home. Base travel time: {travel_time} min. Adding access penalty: {self.home_access_time} min.")
        
        self.traveling = True
        self.travel_time_remaining = travel_time
        self.destination_node = target_node
        self.destination_geometry = target_geometry
        
        # Notify output controller about travel start
        if hasattr(self.model, 'output_controller'):
            self.model.output_controller.track_travel_start(
                agent_id=self.unique_id,
                from_node=self.current_node,
                to_node=target_node,
                travel_time=travel_time
            )
        
        return True

    # =====================================================================
    # 3. MOVEMENT AND TARGET SELECTION MODULE
    # =====================================================================

    def move_to_poi(self, poi_id):
            
        # Find the POI agent with the specified ID
        target_poi = None
        for poi_agent in self.model.poi_agents:
            if poi_agent.unique_id == poi_id:
                target_poi = poi_agent
                break
        
        if target_poi is None:
            if hasattr(self, 'logger'):
                self.logger.warning(f"POI with ID {poi_id} not found")
            return False
        
        # Check if the POI is accessible
        if target_poi.node_id not in self.accessible_nodes:
            if hasattr(self, 'logger'):
                self.logger.warning(f"POI {poi_id} at node {target_poi.node_id} is not accessible")
            return False
        
        # Start traveling to the POI
        return self.start_travel(target_poi.node_id, target_poi.geometry)


    def step(self):
        """Simplified step method using action system"""
        
        
        # Store the initial state of current_action to avoid issues with it being modified during the step
        has_ongoing_action = self.current_action is not None
        try:

            step_count = getattr(self.model, 'step_count', 0)
            self.location_history.append((step_count, self.geometry))
            
            # Check for daily income payment
            current_day = getattr(self.model, 'day_count', 0)
            if current_day > self.last_paid_day:
                self.money += self.daily_income
                self.last_paid_day = current_day
                if hasattr(self, 'logger'):
                    self.logger.info(f"Received daily income: ${self.daily_income:.0f} (Total: ${self.money:.0f})")
            
            # Handle ongoing travel
            if self.traveling:

                if hasattr(self.model, 'output_controller'):
                    self.model.output_controller.track_travel_step(self.unique_id)
                
                self.travel_time_remaining -= 1
                
                if self.travel_time_remaining <= 0:
                    # Removed debug print
                    self.traveling = False
                    # Update last visited node before changing current node
                    self.last_visited_node = self.current_node
                    self.current_node = self.destination_node
                    self.geometry = self.destination_geometry
                    self.visited_pois.append(self.destination_node)
                    
                    # Find the POI agent at the destination
                    visited_poi_agent = None
                    for poi in self.model.poi_agents:
                        if poi.node_id == self.destination_node:
                            visited_poi_agent = poi
                            break
                    
                    if visited_poi_agent:
                        # Add self to the POI's visitor list
                        visited_poi_agent.add_visitor(self.unique_id)
                        
                        # --- MEMORY MODULE: Record visit ---
                        self.memory['visited_pois'].append({
                            'step': getattr(self.model, 'step_count', None),
                            'poi_id': visited_poi_agent.unique_id,
                            'poi_type': visited_poi_agent.poi_type,
                            'category': getattr(visited_poi_agent, 'category', None),
                            'income': self.attributes['income']
                        })
                        
                        # Track POI visit in output controller
                        if hasattr(self.model, 'output_controller'):
                            poi_category = getattr(visited_poi_agent, 'category', 'other')
                            self.model.output_controller.track_poi_visit(poi_category, self.unique_id)
                    
                    # Reset travel attributes
                    self.destination_node = None
                    self.destination_geometry = None
                
                    # Start pending action if we have one
                    if self.pending_action:
                        print(f"Arrived at destination! Starting pending action: {self.pending_action.name}")
                        self.current_action = self.pending_action
                        self.action_time_remaining = self.pending_action.duration_minutes
                        self.pending_action = None  # Clear pending action
                        
                        if hasattr(self, 'logger'):
                            self.logger.info(f"Started pending action '{self.current_action.name}' for {self.current_action.duration_minutes} minutes")

                return
            
            # Handle ongoing action
            if has_ongoing_action:     
                self.action_time_remaining -= 1
                if self.action_time_remaining <= 0:
                    self._complete_action()
                    self.current_action = None
                return
            else:
                self._decide_next_action()
                return  
            
        except Exception as e:
            # Removed debug print
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in resident step: {e}")
            import traceback
            traceback.print_exc()

    # =====================================================================
    # ACTION SYSTEM METHODS
    # =====================================================================
    
    def _decide_next_action(self, excluded_actions=None):
        """Decide the next action to take using the action system."""
        
        # Get current context
        hour = self.model.hour_of_day
        temperature = self.model.temperature if hasattr(self.model, 'temperature') else 25.0
        
        # Get available actions
        available_actions = get_available_actions(
            hour=hour,
            is_employed=self.is_employed,
            money=self.money
        )
        if excluded_actions:
            for excluded in excluded_actions:
                available_actions.pop(excluded, None)
            print(f" Excluded failed actions: {excluded_actions}")
        if not available_actions:
            if hasattr(self, 'logger'):
                self.logger.info(f"No actions available at hour {hour}, defaulting to rest")
            self._start_action(copy.deepcopy(EVERYDAY_ACTIONS["rest"]))
            return

        if self.is_employed and 9 <= hour < 17:
            print(f"WORK MANDATORY: Agent {self.unique_id} (employed) must work at hour {hour}")

        action_name = None
        if self.brain :
            action_name = self._llm_decide_action(available_actions, temperature)
        else:
            # Fallback: choose first available action for non-LLM agents
            action_name = list(available_actions.keys())[0] if available_actions else None
            print(f" agent chose without brain: {action_name} (first available action)")
        if action_name and action_name in available_actions:
            success = self._start_action(copy.deepcopy(available_actions[action_name]))
            if not success and self.current_action is None:
                print(f" Action '{action_name}' failed, retrying with different action...")
                excluded_actions = excluded_actions or []
                excluded_actions.append(action_name)
                self._decide_next_action(excluded_actions)
            return  # Return immediately after starting action
        else:
            self._start_action(copy.deepcopy(EVERYDAY_ACTIONS["rest"]))
            return  # Return immediately after starting action
    
    def _llm_decide_action(self, available_actions: Dict[str, Action], temperature: float) -> Optional[str]:
        
        # Build memory summary
        memory_summary = "No recent actions"
        if self.action_memory:
            recent = self.action_memory[-5:]  # Last 5 actions
            memory_summary = "Recent actions: " + ", ".join([f"{action} at {time}" for action, time in recent])

        observation = (
            f"Current state: Money=${self.money:.0f}, Health={self.health_status}, "
            f"Hour={self.model.hour_of_day}, Outside temperature={temperature:.1f}°C. "
            f"{memory_summary}. "
            f"Available actions: {list(available_actions.keys())}"
        )

        try:
            self.brain.observe(observation)
            action_list = []
            for action_name, action in available_actions.items():
                action_list.append(
                    f"- {action_name}: {action.description} | cost=${action.cost:.1f} | duration={action.duration_minutes} min | location={action.location_type}"
                )
            actions_text = "\n".join(action_list)
            
            # Build demographic profile
            demographic_info = []
            demographic_info.append(f"Age: {self.attributes.get('age', 'unknown')}")
            demographic_info.append(f"Gender: {self.attributes.get('gender', 'unknown')}")
            demographic_info.append(f"Education: {self.attributes.get('education', 'unknown')}")
            demographic_info.append(f"Occupation: {self.attributes.get('occupation', 'unemployed' if not self.attributes.get('occupation') else self.attributes['occupation'])}")
            demographic_info.append(f"Industry: {self.attributes.get('industry', 'none' if not self.attributes.get('industry') else self.attributes['industry'])}")
            demographic_info.append(f"Employment Status: {self.economic_status}")
            demographic_info.append(f"Household Type: {self.household_type}")
            demographic_info.append(f"Parish: {self.attributes.get('parish', 'unknown')}")
            
            demographics_text = ", ".join(demographic_info)
            
            prompt = (
                f"You are a resident of Macau, China with the following profile:\n"
                f"{demographics_text}\n\n"
                f"Current Situation:\n"
                f"Money: ${self.money:.0f}\n"
                f"Health Status: {self.health_status} (your personal health condition)\n"
                f"Current Time: {self.model.hour_of_day}:00\n"
                f"Outside Weather Temperature: {temperature:.1f}°C (this is the outdoor air temperature, not your body temperature)\n"
                f"{memory_summary}\n\n"
                f"You must choose ONE action from the list below, considering your demographic profile and current situation:\n"
                f"AVAILABLE ACTIONS:\n{actions_text}\n\n"
                f"Consider:\n"
                f"- Your education level and occupation when choosing activities\n"
                f"- The cost of each action relative to your available money\n"
                f"- The temperature of the outside weather (especially if it's hot) and the time of day\n"
                f"- Your age and household situation\n"
                f"- Your employment status and work requirements\n\n"
                f"Respond in this exact format:\n"
                f"Action: [action_name]\n"
                f"Reason: [brief explanation why you chose this action based on your profile]\n\n"
                f"Choose ONLY from the actions listed above."
            )
            
            response = self.brain.decide(prompt)
            if response:
                print(f"LLM Full Response:\n{response}")
                
                action_match = re.search(r'Action:\s*([^\n]+)', response, re.IGNORECASE)
                reason_match = re.search(r'Reason:\s*([^\n]+)', response, re.IGNORECASE)
                
                if action_match:
                    action_name = action_match.group(1).strip().lower().replace('"', '').replace("'", "")
                    reason = reason_match.group(1).strip() if reason_match else "No reason provided"

                    if action_name in available_actions:
                        print(f"LLM chose: '{action_name}' | Reason: {reason}")
                        return action_name
                    else:
                        # Try to find a matching action by exact name
                        for available_action in available_actions:
                            if available_action.lower() == action_name:
                                print(f"LLM chose: '{available_action}' (matched from '{action_name}') | Reason: {reason}")
                                return available_action
                        
                        print(f" LLM chose invalid action: '{action_name}' | Reason: {reason}")
                        print(f" Available actions: {list(available_actions.keys())}")
                        return None
                else:
                    # Fallback: try to find any action name in the response
                    response_lower = response.lower()
                    for available_action in available_actions:
                        if available_action in response_lower:
                            print(f"Found '{available_action}' in unstructured response: {response[:100]}...")
                            return available_action
                    
                    print(f" Could not parse action from response: {response[:100]}...")
                    return None
            else:
                print(f" LLM returned empty response")
                return None
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"LLM decision error: {e}")


    def _start_action(self, action: Action):
        """Start executing an action."""
        # Enforce affordability for healthcare
        if action.name in ("visit_doctor", "go_pharmacy") and self.money < float(action.cost):
            # Cannot afford
            self.failure_counts["cost"] += 1
            self.healthcare_access_success = False
            return False

        target_poi = self._find_poi_for_action(action.location_type)
        print(f" POI Search: action='{action.name}', location_type='{action.location_type}', found_poi={target_poi is not None}")
        
        if target_poi and action.location_type != "home":
            action.target_poi_id = target_poi.unique_id
            print(f" POI Found: {target_poi.unique_id} (type: {getattr(target_poi, 'poi_type', 'unknown')})")
            
            # Move to POI if not already there
            if target_poi.node_id != self.current_node:
                print(f"Agent needs to travel to POI {target_poi.unique_id} first")
                # Store the action to start after arrival
                self.pending_action = action
                self.current_action = None  # No current action during travel
                self.move_to_poi(target_poi.unique_id)
                return True  # Travel started successfully
            else:
                print(f"Agent already at POI, starting action immediately")
                self.current_action = action
        elif action.location_type != "home":
            agent_parish = getattr(self, 'parish', 'unknown')
            self.model.track_unmet_demand(agent_parish, action.location_type)
            # If this was a healthcare attempt, mark failure
            if action.name in ("visit_doctor", "go_pharmacy"):
                self.failure_counts["distance"] += 1
                self.healthcare_access_success = False
            self.current_action = None
            return False  # Return False to indicate failure
        else:
            # Do action at current location (home or current node)
            if action.location_type == "home" and self.current_node != self.home_node:
                print(f"Agent needs to travel home first")
                # Store the action to start after arriving home
                self.pending_action = action
                self.current_action = None  # No current action during travel
                self.go_home()
                return True  # Travel started successfully
            else:
                # Already at the right location, start action immediately
                print(f" Agent at correct location, starting action immediately")
                self.current_action = action

        if self.current_action:
            self.action_time_remaining = self.current_action.duration_minutes
            if hasattr(self, 'logger'):
                self.logger.info(f"Started action '{self.current_action.name}' for {self.current_action.duration_minutes} minutes")
        
        return True  # Return True to indicate success
    
    def _complete_action(self):
        """Complete the current action and update state."""
        
        if not self.current_action:
            return
        # Deduct cost when completing the action
        try:
            self.money -= float(self.current_action.cost)
        except Exception:
            pass
        current_time = f"Day {self.model.day_count + 1} {self.model.hour_of_day:02d}:{self.model.step_count % 60:02d}"
        self.action_memory.append((self.current_action.name, current_time))
        # Mark healthcare access success if applicable
        if self.current_action.name in ("visit_doctor", "go_pharmacy"):
            # Only set to True; leave None as-is until first success/failure evaluation
            self.healthcare_access_success = True

    
    def _find_poi_for_action(self, location_type: str) -> Optional[Any]:

        if location_type == "home":
            return None  # Stay at home node
        
        if location_type == "workplace":
            # Find office/workplace POIs, or use random node as fallback
            if hasattr(self.model, 'poi_agents'):
                for poi in self.model.poi_agents:
                    poi_type = getattr(poi, 'poi_type', '').lower()
                    if any(x in poi_type for x in ['office', 'workplace', 'business', 'company']):
                        if poi.node_id in self.accessible_nodes:
                            return poi
            
            # Fallback: use a random accessible node as workplace
            if self.accessible_nodes:
                work_node = random.choice(list(self.accessible_nodes.keys()))
                # Create a proper workplace POI and add it to the model
                from agents.fifteenminutescity.poi import POI
                from shapely.geometry import Point
                
                # Get node coordinates
                node_coords = self.model.graph.nodes[work_node]
                work_geometry = Point(node_coords['x'], node_coords['y'])
                
                # Create workplace POI
                workplace_poi = POI(
                    model=self.model,
                    unique_id=f"workplace_{work_node}",
                    geometry=work_geometry,
                    node_id=work_node,
                    poi_type="office",
                    category="daily_living"
                )
                
                # Add to model's POI agents list so move_to_poi can find it
                self.model.poi_agents.append(workplace_poi)
                return workplace_poi
        
        # For other locations, find matching POI type
        if hasattr(self.model, 'poi_agents'):
            matching_pois = []
            all_poi_types = []  # For debugging
            
            for poi in self.model.poi_agents:
                # Only consider accessible POIs
                if poi.node_id not in self.accessible_nodes:
                    continue
                    
                poi_type = getattr(poi, 'poi_type', '').lower()
                all_poi_types.append(poi_type)  # Collect for debugging
                
                # Logical matching based on location type
                if location_type == "restaurant":
                    if any(x in poi_type for x in ['restaurant', 'cafe', 'food', 'diner', 'bistro']):
                        matching_pois.append(poi)
                        
                elif location_type == "supermarket":
                    if any(x in poi_type for x in ['supermarket', 'grocery', 'market', 'store']):
                        matching_pois.append(poi)
                        
                elif location_type == "park":
                    if any(x in poi_type for x in ['park', 'garden', 'square', 'green']):
                        matching_pois.append(poi)
                        
                elif location_type == "cafe":
                    if any(x in poi_type for x in ['cafe', 'coffee', 'tea', 'bakery']):
                        matching_pois.append(poi)
                        
                elif location_type == "entertainment":
                    if any(x in poi_type for x in ['cinema', 'theater', 'museum', 'entertainment', 'casino']):
                        matching_pois.append(poi)
                elif location_type == "pharmacy":
                    if any(x in poi_type for x in ['pharmacy']):
                        matching_pois.append(poi)

                elif location_type == "hospital":
                    if any(x in poi_type for x in ['hospital']):
                        matching_pois.append(poi)
            
            # Debug logging for POI search
            print(f" POI Search Debug for location_type='{location_type}':")
            print(f"   Available POI types: {set(all_poi_types)}")
            print(f"   Found {len(matching_pois)} matching POIs")
            
            if matching_pois:
                # Choose closest POI if multiple matches
                if len(matching_pois) > 1:
                    best_poi = min(matching_pois, 
                                 key=lambda p: self.accessible_nodes.get(p.node_id, float('inf')))
                    return best_poi
                else:
                    return matching_pois[0]
        
        return None

    pass


    def _select_path_with_llm(self, path_options, from_node, to_node):
        """
        Use LLM to score paths and select the best one.
        Delegates to LLM interaction layer to avoid code duplication.
        
        Args:
            path_options: List of path dictionaries with metadata
            from_node: Starting node ID
            to_node: Destination node ID
            
        Returns:
            Selected path nodes (list) or None if LLM fails
        """
        if  not path_options:
            # Fallback to shortest path when LLM disabled
            return self._fallback_path_selection(path_options)
        
        try:
            # Track statistics when multiple paths are available
            if len(path_options) > 1:
                self.path_selection_stats['total_multi_path_decisions'] += 1
                # Identify the shortest path (by travel time)
                shortest_path_index = min(range(len(path_options)), 
                                        key=lambda i: path_options[i]['travel_time_minutes'])
            else:
                # Only one path available, no choice to make
                return path_options[0]['path_nodes'] if path_options else None
            
            # Check if we have a Concordia brain
            if not hasattr(self, 'brain') or self.brain is None:
                self.path_selection_stats['fallback_decisions'] += 1
                selected_path_nodes = self._fallback_path_selection(path_options)
                # Track if fallback didn't select shortest path
                if selected_path_nodes and len(path_options) > 1:
                    fallback_selected_index = next((i for i, p in enumerate(path_options) 
                                                  if p['path_nodes'] == selected_path_nodes), None)
                    if fallback_selected_index is not None and fallback_selected_index != shortest_path_index:
                        self.path_selection_stats['shortest_path_not_selected'] += 1
                return selected_path_nodes
            

            time_context = f"Time: {self.model.hour_of_day}:00" if hasattr(self.model, 'hour_of_day') else "Time: unknown"
            
            # Get environmental context with temperature recommendations
            env_context = ""
            temp_context = ""
            if hasattr(self.model, 'get_environmental_context'):
                env_data = self.model.get_environmental_context()
                env_context = (
                    f"Environment: {env_data['temperature']:.1f}°C, "
                    f"time period: {env_data['time_period']}. "
                )
                
                # Add temperature context for path selection
                if env_data.get('temperature_recommendations'):
                    recommendations = env_data['temperature_recommendations']
                    temp_context = (
                        f"TEMPERATURE ADVISORY: {recommendations['health_warning']}. "
                        f"Movement advice: {recommendations['movement']}. "
                    )
            
            # Format path options for Concordia with slope and speed limit info
            path_descriptions = []
            for i, path in enumerate(path_options):
                # Base description
                desc_parts = [
                    f"Path {i+1}: {path['distance_meters']}m",
                    f"{path['travel_time_minutes']} min",
                    f"road type: {path['dominant_road_type']}"
                ]
                
                # Add slope information if available
                if path.get('avg_slope') is not None:
                    slope = path['avg_slope']
                    if slope > 5:
                        desc_parts.append(f"uphill {slope}%")
                    elif slope < -5:
                        desc_parts.append(f"downhill {abs(slope)}%")
                    elif slope != 0:
                        desc_parts.append(f"slope {slope}%")
                    else:
                        desc_parts.append("flat")
                
                # Add speed limit if available (indicates car traffic)
                if path.get('max_speed_limit'):
                    desc_parts.append(f"cars up to {path['max_speed_limit']}km/h")
                
                path_desc = ", ".join(desc_parts)
                path_descriptions.append(path_desc)
            
            paths_text = "\n".join(path_descriptions)
            
            # Create observation for Concordia
            observation = (
                f"Agent {self.unique_id} needs to choose a path from node {from_node} to {to_node}. "
                f"Current needs:  {time_context}. {env_context}{temp_context}Age: {self.age}. "
                f"Available paths:\n{paths_text}"
            )
            
            # Step 2: Give observation to Concordia brain
            self.brain.observe(observation)
            
            # Step 3: Ask Concordia to make decision with temperature awareness
            decision_prompt = (
                "Choose the best path by responding with ONLY the path number (1, 2, or 3). "
                "Consider your needs, age, current weather/temperature, time of day, slopes, and car traffic. "
                "For example: steep uphill may be difficult for elderly, high car speeds may be dangerous, "
                "weather conditions may affect slope difficulty. "
            )

            
            decision_prompt += "Respond with just the number, nothing else."
            
            response = self.brain.decide(decision_prompt)
            
            # Step 4: Parse Concordia response
            selected_path_id = self._parse_concordia_path_response(response, len(path_options))
            
            if selected_path_id is not None:
                selected_path = path_options[selected_path_id]
                self.logger.info(f"Concordia selected path {selected_path_id + 1}")
                
                # Track Concordia decision and whether shortest path was selected
                self.path_selection_stats['concordia_decisions'] += 1
                if selected_path_id != shortest_path_index:
                    self.path_selection_stats['shortest_path_not_selected'] += 1
                    self.logger.debug(f"Concordia chose path {selected_path_id + 1} instead of shortest path {shortest_path_index + 1}")
                
                # Validate path_nodes exists and is valid
                if 'path_nodes' in selected_path and isinstance(selected_path['path_nodes'], list):
                    path_nodes = selected_path['path_nodes']
                    if len(path_nodes) >= 2:  # Must have at least start and end node
                        return path_nodes
                    else:
                        print(f"DEBUG: Resident {self.unique_id} - Path too short: {len(path_nodes)} nodes")
                else:
                    print(f"DEBUG: Resident {self.unique_id} - Invalid path_nodes in selected path")
                
                # If path validation failed, use fallback
                self.path_selection_stats['fallback_decisions'] += 1
                selected_path_nodes = self._fallback_path_selection(path_options)
                # Track if fallback didn't select shortest path
                if selected_path_nodes and len(path_options) > 1:
                    fallback_selected_index = next((i for i, p in enumerate(path_options) 
                                                  if p['path_nodes'] == selected_path_nodes), None)
                    if fallback_selected_index is not None and fallback_selected_index != shortest_path_index:
                        self.path_selection_stats['shortest_path_not_selected'] += 1
                return selected_path_nodes
            else:
                print(f"DEBUG: Resident {self.unique_id} - Could not parse Concordia response: {response}")
                self.path_selection_stats['fallback_decisions'] += 1
                selected_path_nodes = self._fallback_path_selection(path_options)
                # Track if fallback didn't select shortest path
                if selected_path_nodes and len(path_options) > 1:
                    fallback_selected_index = next((i for i, p in enumerate(path_options) 
                                                  if p['path_nodes'] == selected_path_nodes), None)
                    if fallback_selected_index is not None and fallback_selected_index != shortest_path_index:
                        self.path_selection_stats['shortest_path_not_selected'] += 1
                return selected_path_nodes
                
        except Exception as e:
            self.logger.error(f"Error in Concordia path selection: {e}")
            if len(path_options) > 1:
                self.path_selection_stats['fallback_decisions'] += 1
                selected_path_nodes = self._fallback_path_selection(path_options)
                # Track if fallback didn't select shortest path
                if selected_path_nodes:
                    fallback_selected_index = next((i for i, p in enumerate(path_options) 
                                                  if p['path_nodes'] == selected_path_nodes), None)
                    if fallback_selected_index is not None and fallback_selected_index != shortest_path_index:
                        self.path_selection_stats['shortest_path_not_selected'] += 1
                return selected_path_nodes
            else:
                return self._fallback_path_selection(path_options)

    def _parse_concordia_path_response(self, response, num_paths):

        try:
            # Clean the response
            response = response.strip()
            
            # Try to extract just the number
            import re
            numbers = re.findall(r'\d+', response)
            
            if numbers:
                path_num = int(numbers[0])  # Take the first number found
                # Convert to 0-based index and validate
                if 1 <= path_num <= num_paths:
                    return path_num - 1  # Convert to 0-based index
            
            # If no valid number found, try to parse common text patterns
            response_lower = response.lower()
            if 'path 1' in response_lower or 'first' in response_lower:
                return 0
            elif 'path 2' in response_lower or 'second' in response_lower:
                return 1 if num_paths > 1 else None
            elif 'path 3' in response_lower or 'third' in response_lower:
                return 2 if num_paths > 2 else None
            elif 'path 4' in response_lower or 'fourth' in response_lower:
                return 3 if num_paths > 3 else None
            
            # If all parsing fails, return None
            return None
            
        except Exception as e:
            self.logger.warning(f"Error parsing Concordia response '{response}': {e}")
            return None

    def _fallback_path_selection(self, path_options):
        if not path_options:
            print(f"DEBUG: Resident {self.unique_id} - No path options available for fallback")
            return None
        
        # Select path with shortest travel time
        shortest_path = min(path_options, key=lambda p: p['travel_time_minutes'])
        return shortest_path['path_nodes']

    def _paths_are_same(self, path1, path2):

        if not path1 or not path2:
            return False
            
        # Compare the actual node lists
        return path1 == path2
            
    def get_parish_info(self):
        
        if not self.parish:
            return None
            
        return {
            "parish_name": self.parish,
            "home_location": (self.geometry.x, self.geometry.y),
            "demographic_info": {
                "age": self.age,
                "age_class": self.attributes['age_class'],
                "gender": self.attributes['gender'],
                "income": self.attributes['income'],
                "education": self.attributes['education'],
                "employment_status": self.employment_status,
                "household_type": self.household_type
            }
        }

    
    def get_path_selection_stats(self):
        """Return the resident's path selection statistics."""
        return self.path_selection_stats.copy()
    
    def get_non_shortest_path_percentage(self):
        if self.path_selection_stats['total_multi_path_decisions'] == 0:
            return None
        
        percentage = (self.path_selection_stats['shortest_path_not_selected'] / 
                     self.path_selection_stats['total_multi_path_decisions']) * 100
        
        return {
            'percentage': percentage,
            'non_shortest_selected': self.path_selection_stats['shortest_path_not_selected'],
            'total_multi_path_decisions': self.path_selection_stats['total_multi_path_decisions'],
            'concordia_decisions': self.path_selection_stats['concordia_decisions'],
            'fallback_decisions': self.path_selection_stats['fallback_decisions']
        }

    def go_home(self):
        # If already at home or traveling, don't start a new journey
        if self.current_node == self.home_node or self.traveling:
            return False
        
        # Get home node coordinates for geometry
        node_coords = self.model.graph.nodes[self.home_node]
        home_geometry = None
        if 'x' in node_coords and 'y' in node_coords:
            from shapely.geometry import Point
            home_geometry = Point(node_coords['x'], node_coords['y'])
        
        # Start traveling home
        return self.start_travel(self.home_node, home_geometry)
