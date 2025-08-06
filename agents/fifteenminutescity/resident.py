from agents.base_person_agent import BaseAgent
import random
import networkx as nx
import logging
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

import re
from brains.concordia_brain import ConcordiaBrain

class ActionType(Enum):
    """Types of actions residents can perform"""
    # Basic actions (always available)
    WAITING = "waiting"
    TRAVELING = "traveling"
    SLEEPING = "sleeping"
    
    # POI-specific actions
    EATING = "eating"
    SHOPPING = "shopping"
    WORKING = "working"
    STUDYING = "studying"
    EXERCISING = "exercising"
    SOCIALIZING = "socializing"
    HEALTHCARE = "healthcare"
    ENTERTAINMENT = "entertainment"
    WORSHIP = "worship"
    BANKING = "banking"
    
    # Granular actions (for detailed simulations)
    TALKING = "talking"
    READING = "reading"
    WATCHING_MOVIE = "watching_movie"
    DOING_BUSINESS = "doing_business"
    BROWSING = "browsing"
    CONSULTING = "consulting"
    PRAYING = "praying"
    PLAYING = "playing"

class ActionGranularity(Enum):
    """Granularity levels for action simulation"""
    SIMPLE = "simple"      # Just waiting times
    BASIC = "basic"        # Basic action types (eating, shopping, etc.)
    DETAILED = "detailed"  # Granular actions (talking, browsing, etc.)

@dataclass
class Action:
    """Represents an action being performed by a resident"""
    action_type: ActionType
    duration: int  # Duration in time steps (minutes)
    poi_type: Optional[str] = None
    poi_id: Optional[int] = None
    description: Optional[str] = None
    needs_satisfied: Optional[Dict[str, int]] = None
    social_interaction: bool = False
    other_agents: Optional[List[int]] = None
    context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.needs_satisfied is None:
            self.needs_satisfied = {}
        if self.other_agents is None:
            self.other_agents = []
        if self.context is None:
            self.context = {}

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
            
            # Behavior and preferences
            'needs_selection': kwargs.get('needs_selection', 'random'),
            'movement_behavior': kwargs.get('movement_behavior', 'need-based'),
            'daily_schedule': kwargs.get('daily_schedule', {}),
            'personality_traits': kwargs.get('personality_traits', {}),
            'activity_preferences': kwargs.get('activity_preferences', {}),
            
            # Physical attributes
            'access_distance': kwargs.get('access_distance', 0),
        }
        
        # Convenience properties for frequently accessed attributes
        self.age = self.attributes['age']
        self.parish = self.attributes['parish']
        self.needs_selection = self.attributes['needs_selection']
        self.movement_behavior = self.attributes['movement_behavior']
        self.social_network = self.attributes['social_network']
        
        # TEMPORARY: Add is_tourist as a direct attribute for easy access
        self.is_tourist = kwargs.get('is_tourist', False)

        
        
        # Dynamic needs (placeholder - to be implemented later)
        self.dynamic_needs = {
            "hunger": 0,
            "social": 0,
            "recreation": 0,
            "shopping": 0,
            "healthcare": 0,
            "education": 0
        }

        # Current needs (will be updated each step)
        self.current_needs = self.dynamic_needs.copy()
        
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
        
        # Action system (replaces simple waiting)
        self.performing_action = False
        self.current_action = None  # Current action being performed
        self.action_time_remaining = 0
        self.action_history = []  # Track completed actions
        
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
        
        # DEBUG: Print comprehensive resident characteristics
        # print(f"DEBUG: Resident {unique_id} characteristics:")
        # print(f"  - Age: {self.attributes['age']} (class: {self.attributes.get('age_class', 'N/A')})")
        # print(f"  - Gender: {self.attributes.get('gender', 'N/A')}")
        # print(f"  - Education: {self.attributes.get('education', 'N/A')}")
        # print(f"  - Occupation: {self.attributes.get('occupation', 'N/A')}")
        # print(f"  - Industry: {self.attributes.get('industry', 'N/A')}")
        # print(f"  - Parish: {self.attributes.get('parish', 'N/A')}")
        # print(f"  - Employment Status: {self.employment_status}")
        # print(f"  - Household Type: {self.household_type}")
        # print(f"  - Monthly Income: {self.attributes['income']}")
        # print(f"  - Daily Income: {self.daily_income:.2f}")
        # print(f"  - Movement Behavior: {self.movement_behavior}")
        # print(f"  - Is Tourist: {self.is_tourist}")
        # print(f"  - Home Node: {self.home_node}")
        # print("---")
        
        # Initialize logger if not provided
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"Resident-{unique_id}")

        # ---------------------------------------------------
        # Concordia Brain (LLM) integration
        # ---------------------------------------------------
        print(f" Agent {unique_id}: Initializing brain...")
        
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
        
        # Social interaction attributes
        self.social_propensity = random.uniform(0.3, 0.8)  # Individual social tendency
        self.contacts = set()  # Set of agent IDs for social connections
        self.interaction_history = []  # List of interaction records

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
        # print(f"DEBUG: _generate_agent_properties called with:")
        # print(f"  - parish: {parish}")
        # print(f"  - city: {city}")
        # print(f"  - industry_distribution available: {industry_distribution is not None}")
        # print(f"  - occupation_distribution available: {occupation_distribution is not None}")
        # print(f"  - income_distribution available: {income_distribution is not None}")
        # print(f"  - economic_status_distribution available: {economic_status_distribution is not None}")
        
        # Use parish-specific demographics if available
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
            if 16 <= age <= 24:
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
    
    def calculate_travel_time(self, from_node, to_node):
        """
        Calculate the travel time between two nodes based on age-class-specific step sizes.
        - Residents with age_class indicating 65+: 60-meter steps (1 minute at 3.6km/h)
        - Younger residents: 80-meter steps (1 minute at 4.8km/h)
        Always rounds up to ensure the agent doesn't move more than their step size per minute.
        
        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            
        Returns:
            Number of time steps needed for travel (each step is 1 minute)
        """
        # For LLM-enabled agents, use path selection instead of simple shortest path
        if self.movement_behavior == 'llms':
            return self._calculate_travel_time_with_path_selection(from_node, to_node)
        
        # Standard shortest path calculation for non-LLM agents
        return self._calculate_standard_travel_time(from_node, to_node)

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
            import itertools
            
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
        print(f"DEBUG: path time calculated!")
        
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
        travel_time = self.calculate_travel_time(self.current_node, target_node)
        
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
        """
        Move to a specific POI by its unique ID.
        
        Args:
            poi_id: The unique ID of the POI agent to move to
            
        Returns:
            Boolean indicating if the move was successful
        """
            
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

    # =====================================================================
    # 4. NEEDS MANAGEMENT MODULE
    # =====================================================================
    
    def get_need_to_poi_mapping(self):
        """
        Map needs to POI types that can satisfy them.
        
        Returns:
            Dictionary mapping need types to lists of POI types
        """
        return {
            "hunger": ["restaurant", "cafe", "fast_food", "food_court", "bar", "pub"],
            "social": ["restaurant", "cafe", "bar", "pub", "community_centre", "place_of_worship", "park"],
            "recreation": ["park", "cinema", "theatre", "sports_centre", "museum", "library", "tourist_attraction", "casino"],
            "shopping": ["shop", "supermarket", "mall", "marketplace", "department_store"],
            "healthcare": ["hospital", "clinic", "pharmacy", "dentist", "doctor"],
            "education": ["school", "university", "college", "library", "training_centre"]
        }

    def find_highest_need(self):
        """
        Find the need with the highest value.
        
        Returns:
            Tuple of (need_type, need_value) for the highest need
        """
        if not self.current_needs:
            return None, 0
        
        highest_need = max(self.current_needs.items(), key=lambda x: x[1])
        return highest_need

    def find_poi_for_need(self, need_type):
        """
        Find available POI types that can satisfy a specific need.
        
        Args:
            need_type: The type of need to satisfy
            
        Returns:
            List of available POI types that can satisfy the need
        """
        need_mapping = self.get_need_to_poi_mapping()
        possible_poi_types = need_mapping.get(need_type, [])
        
        # Filter to only POI types that actually exist in the model
        available_poi_types = []
        
        # Check POI agents first
        if hasattr(self.model, 'poi_agents') and self.model.poi_agents:
            existing_poi_types = set(poi.poi_type for poi in self.model.poi_agents)
            available_poi_types.extend([poi_type for poi_type in possible_poi_types if poi_type in existing_poi_types])
        
        # Fall back to POI dictionary if no POI agents
        if not available_poi_types and hasattr(self.model, 'pois') and self.model.pois:
            existing_poi_types = set(self.model.pois.keys())
            available_poi_types.extend([poi_type for poi_type in possible_poi_types if poi_type in existing_poi_types])
        
        return available_poi_types

    def satisfy_need_at_poi(self, need_type, satisfaction_amount=30):
        """
        Reduce a need when visiting a POI that satisfies it.
        
        Args:
            need_type: The type of need being satisfied
            satisfaction_amount: How much the need is reduced (default: 30)
        """
        if need_type in self.current_needs:
            self.current_needs[need_type] = max(0, self.current_needs[need_type] - satisfaction_amount)

    def choose_movement_target(self):
        """Choose a movement target based on the configured movement behavior."""
        print(f"🎯 Agent {self.unique_id}: choose_movement_target called with movement_behavior='{self.movement_behavior}'")
        
        if self.movement_behavior == 'llms':
            print(f"🤖 Agent {self.unique_id}: Using LLM-based movement")
            return self._choose_llm_based_target()
        elif self.movement_behavior == 'need-based':
            print(f"🎯 Agent {self.unique_id}: Using need-based movement")
            return self._choose_need_based_target()

        else:  # random movement
            target = self._choose_random_target()
            if target is None and self.current_node != self.home_node:
                # No suitable POI available and not at home - force going home
                return 'home'
            return target

    def _choose_need_based_target(self):
        """
        Choose movement target based on current needs.
        
        Returns:
            POI type to move to, or None if no suitable POI found
        """
        
        return None

    def _choose_random_target(self):
        """
        Choose a random POI ID for movement.
        For random movement, prevents visiting the same POI twice in a row.
        If no other POIs are available, returns None to trigger going home.
        
        Returns:
            Random POI ID, or None if no suitable POIs available
        """
        # Get all available POI agents
        available_poi_ids = []
        
        if hasattr(self.model, 'poi_agents') and self.model.poi_agents:
            # Get POI agents and their node locations
            for poi in self.model.poi_agents:
                if poi.node_id in self.accessible_nodes:
                    # Skip if this is the last visited node (prevent consecutive visits)
                    if self.last_visited_node is not None and poi.node_id == self.last_visited_node:
                        continue
                    available_poi_ids.append(poi.unique_id)
        
        if available_poi_ids:
            return random.choice(available_poi_ids)
        
        # No suitable POIs available - return None to trigger going home
        return None

    def _choose_llm_based_target(self):
        """Choose movement target using Concordia LLM brain (if available)."""
        # Removed debug print
        
        # Prefer Concordia brain if present
        if getattr(self, 'brain', None) is not None:
            # Removed debug print
            return self._choose_concordia_based_target()
        else:
            print(f"Agent {self.unique_id}: No brain found"	)
            pass

        # Concordia unavailable – fall back to need-based behaviour
        # Removed debug print
        return self._choose_need_based_target()
    

    
    def _parse_llm_decision(self, decision):
        """
        Parse LLM decision to extract movement target.
        
        Args:
            decision: LLMDecision object
            
        Returns:
            POI type to move to, 'home', or None
        """
        try:
            # Check if decision has the expected action attribute
            if not hasattr(decision, 'action') or decision.action is None:
                self.logger.warning(f"Decision object missing action attribute: {decision}")
                return None
            
            action = str(decision.action).lower()
            
            # Check if the action is to go home
            if 'home' in action or 'return' in action:
                return 'home'
            
            # Check if the action is to wait/stay
            if 'wait' in action or 'stay' in action or 'rest' in action:
                return None
            
            # Try to extract POI type from the action
            poi_types = ['restaurant', 'cafe', 'shop', 'hospital', 'school', 'park', 
                        'library', 'cinema', 'gym', 'pharmacy', 'bank', 'supermarket']
            
            for poi_type in poi_types:
                if poi_type in action:
                    return poi_type
            
            # If no specific POI type found, try to infer from action keywords
            if 'eat' in action or 'food' in action or 'hungry' in action:
                return 'restaurant'
            elif 'shop' in action or 'buy' in action:
                return 'shop'
            elif 'health' in action or 'medical' in action:
                return 'hospital'
            elif 'exercise' in action or 'fitness' in action:
                return 'gym'
            elif 'relax' in action or 'nature' in action:
                return 'park'
            
            # Default: return None to stay put
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM decision: {e}, decision: {decision}")
            return None

    def _satisfy_needs_at_poi(self, poi_type):
        """
        Satisfy needs when visiting a POI of a specific type.
        
        Args:
            poi_type: The type of POI being visited
        """
        need_mapping = self.get_need_to_poi_mapping()
        
        # Find which needs this POI type can satisfy
        for need_type, poi_types in need_mapping.items():
            if poi_type in poi_types:
                # Satisfy this need
                satisfaction_amount = random.randint(20, 40)  # Random satisfaction between 20-40
                self.satisfy_need_at_poi(need_type, satisfaction_amount)

    def increase_needs_over_time(self):
        """
        Gradually increase needs over time to simulate natural need accumulation.
        With 1-minute time steps, needs increase more slowly.
        """
        # Only increase needs every 15 minutes (every 15 steps) to avoid too rapid changes
        if self.model.step_count % 15 == 0:
            for need_type in self.current_needs:
                # Increase each need by a small random amount (1-3 points per 15 minutes)
                increase = random.randint(1, 3)
                self.current_needs[need_type] = min(100, self.current_needs[need_type] + increase)

    def step(self):
        """Advance the agent one step"""
        # Removed debug print
        
        try:
            super().step()
            # Removed debug print
            
            # Increase needs over time
            self.increase_needs_over_time()
            # Removed debug print
            
            # Handle ongoing travel
            if self.traveling:
                # Removed debug print
                # Track actual travel time in output controller
                if hasattr(self.model, 'output_controller'):
                    self.model.output_controller.track_travel_step(self.unique_id)
                
                self.travel_time_remaining -= 1
                
                # Check if we've arrived
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
                        
                        # Start waiting at POI if it has waiting time
                        if hasattr(visited_poi_agent, 'get_waiting_time'):
                            waiting_time = visited_poi_agent.get_waiting_time()
                            if waiting_time > 0:
                                # Determine action based on POI type and simulation settings
                                action = self._select_action_at_poi(visited_poi_agent, waiting_time)
                                self._start_action(action)
                                
                                # Track waiting time in output controller
                                if hasattr(self.model, 'output_controller'):
                                    self.model.output_controller.track_waiting_start(self.unique_id, poi_category, waiting_time)
                        
                        # Satisfy needs if we're using need-based movement
                        if self.movement_behavior == 'need-based':
                            self._satisfy_needs_at_poi(visited_poi_agent.poi_type)
                        
                        # Update emotional state if LLM behavior is enabled
                        if hasattr(self, 'emotional_state') and hasattr(self.model, 'persona_memory_manager'):
                            self._update_emotional_state_from_poi_visit(visited_poi_agent.poi_type, visited_poi_agent)
                    
                    # Reset travel attributes
                    self.destination_node = None
                    self.destination_geometry = None
                
                # Still traveling, don't take any other movement actions
                # Removed debug print
                return
            
            # Handle ongoing actions at POIs
            if self.performing_action:
                # Removed debug print
                self.action_time_remaining -= 1
                
                # While performing an action, check for social interactions
                self._check_for_social_interaction()
                
                # Check if action is finished
                if self.action_time_remaining <= 0:
                    self._complete_current_action()
                
                # Still performing action, don't take any other movement actions
                # Removed debug print
                return
            
            # === MOVEMENT DECISION MAKING ===
            # This is where all movement decisions are made based on the movement behavior
            if not self.traveling:
                # Removed debug print
                target_poi = None
                
                if self.movement_behavior == 'random':
                    # Removed debug print
                    # Random movement - use existing simple logic
                    target_poi = self._make_random_movement_decision()
                    
                elif self.movement_behavior == 'need-based':
                    # Removed debug print
                    # Need-based movement - use existing logic but centralized here
                    target_poi = self._make_need_based_movement_decision()
                    
                elif self.movement_behavior == 'llms':
                    # Removed debug print
                    # LLM-based movement - placeholder for future sophisticated decision making
                    target_poi = self._make_llm_movement_decision()
                else:
                    # Removed debug print
                    target_poi = self._make_need_based_movement_decision()
                
                # Removed debug print
                
                # Execute the movement decision
                if target_poi == 'home':
                    # Removed debug print
                    self.go_home()
                elif target_poi:
                    # Removed debug print
                    self.move_to_poi(target_poi)
                else:
                    # Removed debug print
                    pass
                                 # If target_poi_type is None, resident stays put this step
            

            
            # Removed debug print
            
        except Exception as e:
            # Removed debug print
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in resident step: {e}")
            import traceback
            traceback.print_exc()



        # ===============================================Action module===============================================


    def _select_action_at_poi(self, poi_agent, waiting_time):
        """
        Select an appropriate action to perform at a POI based on the POI type and simulation granularity.
        
        Args:
            poi_agent: The POI agent where the action will be performed
            waiting_time: Duration in minutes for the action
            
        Returns:
            Action object representing what the resident will do at the POI
        """
        poi_type = poi_agent.poi_type
        
        # Map POI types to appropriate actions based on granularity
        if self.model.action_granularity == ActionGranularity.SIMPLE:
            # Simple: just waiting
            action_type = ActionType.WAITING
            description = f"Waiting at {poi_type}"
            
        elif self.model.action_granularity == ActionGranularity.BASIC:
            # Basic: map POI types to basic action types
            poi_to_action = {
                'restaurant': ActionType.EATING,
                'cafe': ActionType.EATING,
                'bar': ActionType.SOCIALIZING,
                'pub': ActionType.SOCIALIZING,
                'shop': ActionType.SHOPPING,
                'supermarket': ActionType.SHOPPING,
                'mall': ActionType.SHOPPING,
                'hospital': ActionType.HEALTHCARE,
                'clinic': ActionType.HEALTHCARE,
                'pharmacy': ActionType.HEALTHCARE,
                'library': ActionType.STUDYING,
                'school': ActionType.STUDYING,
                'university': ActionType.STUDYING,
                'gym': ActionType.EXERCISING,
                'sports_centre': ActionType.EXERCISING,
                'park': ActionType.SOCIALIZING,
                'cinema': ActionType.ENTERTAINMENT,
                'theatre': ActionType.ENTERTAINMENT,
                'bank': ActionType.BANKING,
                'church': ActionType.WORSHIP,
                'mosque': ActionType.WORSHIP,
                'temple': ActionType.WORSHIP,
                'community_centre': ActionType.SOCIALIZING
            }
            action_type = poi_to_action.get(poi_type, ActionType.WAITING)
            description = f"{action_type.value.title()} at {poi_type}"
            
        else:  # DETAILED granularity
            # Detailed: more specific actions based on POI type
            poi_to_detailed_action = {
                'restaurant': ActionType.EATING,
                'cafe': ActionType.TALKING,
                'bar': ActionType.SOCIALIZING,
                'pub': ActionType.SOCIALIZING,
                'shop': ActionType.BROWSING,
                'supermarket': ActionType.SHOPPING,
                'mall': ActionType.BROWSING,
                'hospital': ActionType.CONSULTING,
                'clinic': ActionType.CONSULTING,
                'pharmacy': ActionType.DOING_BUSINESS,
                'library': ActionType.READING,
                'school': ActionType.STUDYING,
                'university': ActionType.STUDYING,
                'gym': ActionType.EXERCISING,
                'sports_centre': ActionType.PLAYING,
                'park': ActionType.SOCIALIZING,
                'cinema': ActionType.WATCHING_MOVIE,
                'theatre': ActionType.ENTERTAINMENT,
                'bank': ActionType.DOING_BUSINESS,
                'church': ActionType.PRAYING,
                'mosque': ActionType.PRAYING,
                'temple': ActionType.PRAYING,
                'community_centre': ActionType.TALKING
            }
            action_type = poi_to_detailed_action.get(poi_type, ActionType.WAITING)
            description = f"{action_type.value.title()} at {poi_type}"
        
        # Determine which needs this action might satisfy
        needs_satisfied = {}
        if action_type in [ActionType.EATING]:
            needs_satisfied['food'] = random.randint(20, 40)
        elif action_type in [ActionType.SOCIALIZING, ActionType.TALKING]:
            needs_satisfied['social'] = random.randint(15, 30)
        elif action_type in [ActionType.EXERCISING, ActionType.PLAYING]:
            needs_satisfied['health'] = random.randint(10, 25)
            needs_satisfied['recreation'] = random.randint(15, 30)
        elif action_type in [ActionType.SHOPPING, ActionType.BROWSING]:
            needs_satisfied['shopping'] = random.randint(20, 35)
        elif action_type in [ActionType.HEALTHCARE, ActionType.CONSULTING]:
            needs_satisfied['health'] = random.randint(25, 50)
        elif action_type in [ActionType.ENTERTAINMENT, ActionType.WATCHING_MOVIE]:
            needs_satisfied['recreation'] = random.randint(20, 40)
        elif action_type in [ActionType.STUDYING, ActionType.READING]:
            needs_satisfied['education'] = random.randint(15, 30)
        elif action_type in [ActionType.WORSHIP, ActionType.PRAYING]:
            needs_satisfied['spiritual'] = random.randint(20, 40)
        elif action_type in [ActionType.BANKING, ActionType.DOING_BUSINESS]:
            needs_satisfied['financial'] = random.randint(15, 25)
        
        # Create and return the action
        return Action(
            action_type=action_type,
            duration=waiting_time,
            poi_type=poi_type,
            poi_id=poi_agent.unique_id,
            description=description,
            needs_satisfied=needs_satisfied,
            social_interaction=(action_type in [ActionType.SOCIALIZING, ActionType.TALKING]),
            context={'poi_category': getattr(poi_agent, 'category', poi_type)}
        )

    def _start_action(self, action: Action):
        """Begin an action at a POI."""
        self.performing_action = True
        self.current_action = action
        self.action_time_remaining = action.duration
        self.action_history.append(action)

    def _complete_current_action(self):
        """Complete the current action and clean up."""
        if self.current_action and self.current_action.poi_id:
            # Find the POI agent to notify them of departure
            poi_agent = self.model.get_agent_by_id(self.current_action.poi_id)
            if poi_agent:
                poi_agent.remove_visitor(self.unique_id)
        
        self.performing_action = False
        self.current_action = None
        self.action_time_remaining = 0

    def _get_poi_socialness_factor(self, poi_type: str) -> float:
        """Get a factor representing how 'social' a POI type is."""
        social_pois = {
            # High socialness
            "cafe": 1.5, "bar": 1.5, "pub": 1.5, "community_centre": 1.5, "park": 1.4,
            # Medium socialness
            "restaurant": 1.2, "library": 1.1, "gym": 1.1, "sports_centre": 1.1,
            # Low socialness
            "shop": 0.8, "supermarket": 0.7, "mall": 0.8,
            # Very low socialness
            "hospital": 0.5, "clinic": 0.5, "pharmacy": 0.4, "bank": 0.6
        }
        return social_pois.get(poi_type, 1.0) # Default to 1.0

    def _check_for_social_interaction(self):
        """Check for and execute social interactions with co-located agents."""
        if not self.performing_action or not self.current_action.poi_id:
            return

        # Get the POI agent where the action is happening
        poi_agent = self.model.get_agent_by_id(self.current_action.poi_id)
        if not poi_agent or len(poi_agent.visitors) < 2:
            return

        # Don't check at every single step. Chance to check is proportional to social_propensity.
        if random.random() > (self.social_propensity * 0.1): # e.g., 5% chance per step for 0.5 propensity
            return

        # Check for interactions with other visitors
        for other_agent_id in poi_agent.visitors:
            if other_agent_id == self.unique_id:
                continue

            other_agent = self.model.get_agent_by_id(other_agent_id)
            if not other_agent or not isinstance(other_agent, Resident):
                continue

            # --- Calculate Interaction Score ---
            # 1. Base score from POI socialness
            interaction_score = self._get_poi_socialness_factor(poi_agent.poi_type)

            # 2. Boost score if already in contacts
            if other_agent_id in self.contacts:
                interaction_score *= 5.0
            
            # 3. Boost score based on homophily (age similarity)
            age_difference = abs(self.age - other_agent.age)
            age_factor = max(0, 1 - (age_difference / 20)) # 1.0 if same age, 0.0 if 20+ years apart
            interaction_score *= (1 + age_factor)

            # 4. Factor in this agent's social propensity
            interaction_score *= self.social_propensity

            # --- Decide if Interaction Happens ---
            # Interaction happens if score > random threshold (e.g., 1.5)
            if interaction_score > (random.random() * 5 + 1.0): # Threshold between 1.0 and 6.0
                self._execute_social_interaction(other_agent)
                # Only one interaction per check to keep it simple
                break
    
    def _execute_social_interaction(self, other_agent):
        """Execute the outcomes of a social interaction."""
        self.logger.info(f"Agent {self.unique_id} is interacting with {other_agent.unique_id}")

        # Increment the model's interaction counter
        # Each agent involved increments it by 0.5 to make 1 full interaction.
        self.model.interactions_this_step += 0.5

        # 1. Record the interaction
        timestamp = self.model.step_count
        interaction = {'type': 'social', 'with': other_agent.unique_id, 'timestamp': timestamp}
        self.interaction_history.append(interaction)
        other_agent.interaction_history.append({'type': 'social', 'with': self.unique_id, 'timestamp': timestamp})

        # 2. Update social needs (decrease them)
        satisfaction = random.randint(10, 25)
        self.satisfy_need_at_poi('social', satisfaction)
        other_agent.satisfy_need_at_poi('social', satisfaction)

        # 3. Form a new connection if they are strangers
        if other_agent.unique_id not in self.contacts:
            if random.random() < 0.25: # 25% chance to form a lasting connection
                self.contacts.add(other_agent.unique_id)
                other_agent.contacts.add(self.unique_id)
                self.logger.info(f"New social connection formed between {self.unique_id} and {other_agent.unique_id}")





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
        if not self.model.llm_enabled or not path_options:
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
                print(f"DEBUG: Resident {self.unique_id} - No Concordia brain available, using fallback")
                self.path_selection_stats['fallback_decisions'] += 1
                selected_path_nodes = self._fallback_path_selection(path_options)
                # Track if fallback didn't select shortest path
                if selected_path_nodes and len(path_options) > 1:
                    fallback_selected_index = next((i for i, p in enumerate(path_options) 
                                                  if p['path_nodes'] == selected_path_nodes), None)
                    if fallback_selected_index is not None and fallback_selected_index != shortest_path_index:
                        self.path_selection_stats['shortest_path_not_selected'] += 1
                return selected_path_nodes
            
            # Step 1: Prepare observation for Concordia brain
            needs_summary = ", ".join(f"{k}:{v}" for k, v in self.current_needs.items())
            time_context = f"Time: {self.model.hour_of_day}:00" if hasattr(self.model, 'hour_of_day') else "Time: unknown"
            
            # Get environmental context with temperature recommendations
            env_context = ""
            temp_context = ""
            if hasattr(self.model, 'get_environmental_context'):
                env_data = self.model.get_environmental_context()
                env_context = (
                    f"Environment: {env_data['weather_description']} ({env_data['temperature']}°C), "
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
                f"Current needs: {needs_summary}. {time_context}. {env_context}{temp_context}Age: {self.age}. "
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
            
            # Add temperature-specific path selection advice
            if env_data.get('temperature_recommendations'):
                recommendations = env_data['temperature_recommendations']
                temp_advice = (
                    f"TEMPERATURE CONSIDERATIONS: Current temperature is {env_data['temperature']:.1f}°C. "
                    f"Health warning: {recommendations['health_warning']}. "
                    f"Movement advice: {recommendations['movement']}. "
                    f"Consider shorter routes if temperature is high to minimize heat exposure. "
                    f"Avoid steep uphill paths in extreme heat. "
                )
                decision_prompt += temp_advice
            
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
        """
        Parse Concordia brain's response to extract path selection.
        
        Args:
            response: Raw response from Concordia brain
            num_paths: Number of available path options
            
        Returns:
            Selected path index (0-based) or None if parsing fails
        """
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
        """
        Fallback path selection - choose shortest time.
        
        Args:
            path_options: Available path options
            
        Returns:
            Path nodes of shortest time path or None
        """
        if not path_options:
            print(f"DEBUG: Resident {self.unique_id} - No path options available for fallback")
            return None
        
        # Select path with shortest travel time
        shortest_path = min(path_options, key=lambda p: p['travel_time_minutes'])
        return shortest_path['path_nodes']

    def _paths_are_same(self, path1, path2):
        """
        Check if two paths are essentially the same.
        
        Args:
            path1: First path (list of node IDs)
            path2: Second path (list of node IDs)
            
        Returns:
            Boolean indicating if paths are the same
        """
        if not path1 or not path2:
            return False
            
        # Compare the actual node lists
        return path1 == path2

    
    def generate_needs(self, method=None):
        """
        Generate needs for the resident based on the specified method.
        
        Args:
            method: The method to use for generating needs. If None, uses self.needs_selection.
                   Options: 'random', 'maslow', 'capability', 'llms'
                   
        Returns:
            Dictionary of generated needs
        """
        if method is None:
            method = self.needs_selection
        
        if method == 'random':
            return self._generate_needs_random()
        elif method == 'maslow':
            return self._generate_needs_maslow()
        elif method == 'capability':
            return self._generate_needs_capability()
        elif method == 'llms':
            return self._generate_needs_llms()
        else:
            # Default to random if unknown method
            return self._generate_needs_random()
    
    def _generate_needs_random(self):
        """
        Generate needs randomly.
        
        Returns:
            Dictionary of randomly generated needs
        """
        needs = {}
        for need_type in self.dynamic_needs.keys():
            # Generate random need level between 0 and 100
            needs[need_type] = random.randint(0, 100)
        
        return needs
    
    def _generate_needs_maslow(self):
        """
        Generate needs based on Maslow's hierarchy of needs.
        Prioritizes basic needs first, then higher-level needs.
        
        Returns:
            Dictionary of needs based on Maslow's hierarchy
        """
        needs = {}

        
        return needs
    
    def _generate_needs_capability(self):
        """
        Generate needs based on capability approach (Sen/Nussbaum).
        Focuses on what people are able to do and be.
        
        Returns:
            Dictionary of needs based on capability approach
        """
        needs = {}

        
        return needs
    
    def _generate_needs_llms(self):
        """
        Generate needs using LLM-inspired approach.
        Uses persona template and emotional state for more realistic need generation.
        
        Returns:
            Dictionary of needs based on LLM-inspired reasoning
        """
        needs = {}
        
        # Initialize with base needs
        for need_type in self.dynamic_needs.keys():
            needs[need_type] = 0
        
        # Check if persona components are available
        if not hasattr(self, 'persona_template') or not hasattr(self, 'emotional_state'):
            # Fall back to random generation if persona not available
            return self._generate_needs_random()
        
        try:
            # Get persona-specific needs based on persona type
            persona_needs = self._get_persona_based_needs()
            
            # Adjust needs based on emotional state
            emotional_adjustments = self._get_emotional_need_adjustments()
            
            # Combine persona needs with emotional adjustments
            for need_type in needs.keys():
                base_need = persona_needs.get(need_type, 30)  # Default moderate need
                emotional_modifier = emotional_adjustments.get(need_type, 1.0)
                
                # Apply emotional modifier
                adjusted_need = base_need * emotional_modifier
                
                # Add some randomness for variability
                random_factor = random.uniform(0.8, 1.2)
                final_need = adjusted_need * random_factor
                
                # Clamp to valid range
                needs[need_type] = max(0, min(100, int(final_need)))
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in LLM needs generation: {e}")
            # Fall back to random generation
            return self._generate_needs_random()
        
        return needs
    
    def _get_persona_based_needs(self):
        """
        Get base needs based on persona type.
        
        Returns:
            Dictionary of base needs for the persona
        """
        # Removed PersonaType import - using dynamic persona generation now
        
        # Generate needs based on simple persona if available
        persona = getattr(self, 'persona', None)
        
        # Default needs
        base_needs = {
            "hunger": 40,
            "social": 35,
            "recreation": 30,
            "shopping": 25,
            "healthcare": 20,
            "education": 15
        }
        
        if persona:
            # Simple needs based on three basic characteristics
            age = persona.age
            household_type = persona.household_type  # 'single', 'family', 'elderly'
            economic_status = persona.economic_status  # 'low', 'middle', 'high'
            
            # Age-based adjustments (simple and interpretable)
            if household_type == 'elderly':
                base_needs.update({
                    "healthcare": 60,  # Higher healthcare needs
                    "social": 45,      # Higher social needs
                    "recreation": 25,  # Lower recreation needs
                    "shopping": 35,    # Regular shopping needs
                    "hunger": 45,      # Regular hunger needs
                    "education": 10    # Lower education needs
                })
            elif household_type == 'family':
                base_needs.update({
                    "shopping": 50,    # Higher shopping needs (family)
                    "healthcare": 35,  # Moderate healthcare needs
                    "social": 30,      # Lower social needs (busy)
                    "recreation": 20,  # Lower recreation needs
                    "hunger": 50,      # Higher hunger needs
                    "education": 25    # Moderate education needs (children)
                })
            else:  # 'single'
                if age < 25:
                    base_needs.update({
                        "education": 50,   # Higher education needs
                        "social": 50,      # Higher social needs
                        "recreation": 40,  # Higher recreation needs
                        "shopping": 25,    # Moderate shopping needs
                        "healthcare": 20,  # Lower healthcare needs
                        "hunger": 45       # Regular hunger needs
                    })
                else:  # Working adults
                    base_needs.update({
                        "recreation": 45,  # Higher recreation needs
                        "social": 40,      # Higher social needs
                        "shopping": 35,    # Moderate shopping needs
                        "healthcare": 25,  # Lower healthcare needs
                        "hunger": 40,      # Regular hunger needs
                        "education": 30    # Moderate education needs
                    })
            
            # Economic status adjustments (simple)
            if economic_status == 'low':
                base_needs["shopping"] = max(15, base_needs["shopping"] - 10)  # Reduced shopping
                base_needs["healthcare"] = max(15, base_needs["healthcare"] - 5)  # Reduced healthcare
            elif economic_status == 'high':
                base_needs["recreation"] = min(60, base_needs["recreation"] + 10)  # More recreation
                base_needs["healthcare"] = min(70, base_needs["healthcare"] + 10)  # Better healthcare
        
        return base_needs
    
    def _get_emotional_need_adjustments(self):
        """
        Get need adjustments based on current emotional state.
        
        Returns:
            Dictionary of multipliers for each need type
        """
        adjustments = {
            "hunger": 1.0,
            "social": 1.0,
            "recreation": 1.0,
            "shopping": 1.0,
            "healthcare": 1.0,
            "education": 1.0
        }
        
        if not hasattr(self, 'emotional_state'):
            return adjustments
        
        try:
            # Get dominant emotion and stress level
            emotions = self.emotional_state.current_emotions
            stress_level = self.emotional_state.stress_level
            
            # Find dominant emotion
            if emotions:
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                
                # Adjust needs based on dominant emotion
                from .persona_memory_modules import EmotionalState
                
                if dominant_emotion == EmotionalState.STRESSED:
                    adjustments["recreation"] *= 1.3  # More recreation when stressed
                    adjustments["social"] *= 0.8      # Less social when stressed
                    adjustments["healthcare"] *= 1.2  # More healthcare when stressed
                elif dominant_emotion == EmotionalState.ANXIOUS:
                    adjustments["healthcare"] *= 1.4  # More healthcare when anxious
                    adjustments["social"] *= 1.2      # More social support when anxious
                elif dominant_emotion == EmotionalState.FRUSTRATED:
                    adjustments["recreation"] *= 1.2  # More recreation when frustrated
                    adjustments["shopping"] *= 1.1    # Slight increase in shopping (retail therapy)
                elif dominant_emotion == EmotionalState.SATISFIED:
                    adjustments["recreation"] *= 0.9  # Less urgent recreation when satisfied
                    adjustments["social"] *= 1.1      # More social when satisfied
                elif dominant_emotion == EmotionalState.WORRIED:
                    adjustments["healthcare"] *= 1.3  # More healthcare when worried
                    adjustments["social"] *= 1.2      # More social support when worried
            
            # Adjust based on stress level
            if stress_level > 0.7:  # High stress
                adjustments["recreation"] *= 1.2
                adjustments["healthcare"] *= 1.1
            elif stress_level < 0.3:  # Low stress
                adjustments["recreation"] *= 0.9
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in emotional need adjustments: {e}")
        
        return adjustments
            
    def get_parish_info(self):
        """
        Get information about the agent's parish.
        
        Returns:
            Dictionary with parish details or None if no parish assigned
        """
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
        """
        Calculate the percentage of times this resident did not select the shortest path.
        
        Returns:
            Dictionary with percentage and counts, or None if no multi-path decisions made
        """
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
        """
        Move to the home node.
        
        Returns:
            Boolean indicating if the move home was successful
        """
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

    def _update_emotional_state_from_poi_visit(self, poi_type, poi_agent):
        """
        Update emotional state based on a POI visit.
        
        Args:
            poi_type: The type of POI visited
            poi_agent: The POI agent visited
        """
        try:
            # Determine satisfaction based on POI type and waiting time
            base_satisfaction = 0.7  # Default satisfaction
            
            # Adjust satisfaction based on POI type
            poi_satisfaction_map = {
                'restaurant': 0.8,
                'cafe': 0.7,
                'park': 0.8,
                'hospital': 0.6,  # Lower satisfaction for healthcare visits
                'shop': 0.7,
                'supermarket': 0.6,
                'library': 0.8,
                'cinema': 0.9,
                'gym': 0.8,
                'school': 0.7
            }
            
            base_satisfaction = poi_satisfaction_map.get(poi_type, 0.7)
            
            # Adjust satisfaction based on waiting time
            if poi_agent and hasattr(poi_agent, 'get_waiting_time'):
                waiting_time = poi_agent.get_waiting_time()
                if waiting_time > 30:  # Long wait reduces satisfaction
                    base_satisfaction *= 0.7
                elif waiting_time > 15:  # Moderate wait slightly reduces satisfaction
                    base_satisfaction *= 0.9
            
            # Create experience for emotional state update
            experience = {
                'type': 'healthcare_visit' if poi_type in ['hospital', 'clinic', 'pharmacy'] else 'service_interaction',
                'outcome': 'positive' if base_satisfaction > 0.6 else 'neutral',
                'satisfaction': base_satisfaction,
                'details': f"Visited {poi_type}"
            }
            
            # Update emotional state through persona memory manager
            self.model.persona_memory_manager.update_agent_experience(str(self.unique_id), experience)
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error updating emotional state from POI visit: {e}")






    def _choose_concordia_based_target(self):
        """Use the embedded Concordia brain to select a movement target.

        New behaviour: ask LLM for JSON output::

            {"action": "move", "target_poi_id": 123}

        action 可取 "move", "home", "stay"。当 action=="move" 时必须给出 target_poi_id。
        如果解析或校验失败，则回退到旧的 need-based 逻辑。
        """

        import json

        # Debug tracing for agent 0 only
        
        
        
        # print(f"\n🤖 CONCORDIA DECISION MAKING TRACE FOR AGENT {self.unique_id}")
        # print("=" * 60)
        # print(f"Agent Current State:")
        # print(f"  - Current Node: {self.current_node}")
        # print(f"  - Persona: {getattr(self, 'persona', None).name if hasattr(self, 'persona') and self.persona else 'N/A'}")
        # print(f"  - Current Needs: {dict(self.current_needs)}")
        # if hasattr(self, 'emotional_state'):
        #     print(f"  - Stress Level: {self.emotional_state.stress_level:.2f}")
        #     dominant_emotions = sorted(self.emotional_state.current_emotions.items(), 
        #                                  key=lambda x: x[1], reverse=True)[:2]
        #     print(f"  - Dominant Emotions: {[(e.value, round(v,2)) for e,v in dominant_emotions]}")

        # -------- 1) 准备可达 POI 列表 --------
        accessible_info = []
        try:
            for poi in getattr(self.model, 'poi_agents', []):
                if poi.node_id not in self.accessible_nodes:
                    continue

                # 估算旅行时间（分钟）；若失败则用 None
                try:
                    # Use standard calculation for POI evaluation to avoid infinite loop
                    # LLM path selection will be used later during actual travel
                    ttime = self._calculate_standard_travel_time(self.current_node, poi.node_id)
                except Exception:
                    ttime = None

                accessible_info.append({
                    "id": poi.unique_id,
                    "type": getattr(poi, 'poi_type', 'unknown'),
                    "travel_time": ttime,
                })
        except Exception as _e:
            self.logger.warning(f"Error building accessible POI info: {_e}")

        # 只保留最近的 20 个（按 travel_time 升序）以避免 prompt 太长
        accessible_info = sorted(accessible_info, key=lambda x: (x["travel_time"] or 1e9))[:20]

        # -------- 2) 构造 observation --------
        needs_summary = ", ".join(f"{k}:{v}" for k, v in self.current_needs.items())
        
        # Get environmental context with temperature recommendations
        env_context = ""
        temp_recommendations = ""
        if hasattr(self.model, 'get_environmental_context'):
            env_data = self.model.get_environmental_context()
            env_context = (
                f"Environment: {env_data['weather_description']} ({env_data['temperature']}°C), "
                f"time period: {env_data['time_period']}. "
            )
            
            # Add temperature-based behavioral recommendations
            if env_data.get('temperature_recommendations'):
                recommendations = env_data['temperature_recommendations']
                temp_recommendations = (
                    f"TEMPERATURE ADVISORY: {recommendations['health_warning']}. "
                    f"Movement: {recommendations['movement']}. "
                    f"Activity level: {recommendations['activity_level']}. "
                    f"Preferred locations: {recommendations['preferred_locations']}. "
                    f"Time preferences: {recommendations['time_preferences']}. "
                )
        
        observation = (
            "Current needs => " + needs_summary + ". "
            + env_context + temp_recommendations +
            "Accessible POIs (first 20) => " + json.dumps(accessible_info) + "."
        )

        
        # print(f"\n📝 Observation sent to Concordia Brain:")
        # print(f"  -> Length: {len(observation)} characters")
        # print(f"  -> Content: {observation[:300]}...")

        # -------- 3) 与 LLM 交互 --------
        try:
            self.brain.observe(observation)
            # Enhanced decision prompt with temperature recommendations
            base_prompt = (
                "Decide your next movement considering current environmental conditions. "
                "Respond STRICTLY in JSON with keys: "
                "'action' (move|home|stay) and, if action=='move', 'target_poi_id' (integer). "
                "Consider weather, temperature, and time of day in your decision. "
            )
            
            # Add temperature-specific decision guidance
            if env_data.get('temperature_recommendations'):
                recommendations = env_data['temperature_recommendations']
                temp_guidance = (
                    f"IMPORTANT: Current temperature is {env_data['temperature']:.1f}°C. "
                    f"Health warning: {recommendations['health_warning']}. "
                    f"Movement advice: {recommendations['movement']}. "
                    f"Preferred locations: {recommendations['preferred_locations']}. "
                    f"Consider these factors in your decision. "
                )
                base_prompt += temp_guidance
            
            base_prompt += "Do NOT output anything except the JSON object."
            
            
            # print(f"\n💭 Decision prompt sent to brain:")
            # print(f"  -> Prompt: {base_prompt[:200]}...")
            # print(f"\n🎯 Asking brain to decide...")
            
            reply = self.brain.decide(base_prompt)
            
            
            # print(f"\n🤖 Brain Response:")
            # print(f"  -> Raw reply: {reply}")
                
        except Exception as _e:
            self.logger.error(f"Concordia brain error: {_e}")
            return self._choose_need_based_target()

        if not reply:
            return None

        # -------- 4) 解析 JSON --------
        try:
            decision = json.loads(reply)
            
            print(f"\n✅ Successfully parsed JSON decision:")
            print(f"  -> Parsed decision: {decision}")
        except Exception as _e:
            self.logger.warning(f"LLM reply is not valid JSON: {reply} / {_e}")
            return self._choose_need_based_target()

        action = str(decision.get("action", "")).lower()
        if action == "home":
            return 'home'
        if action in ("stay", "wait", "rest"):
            return None

        if action == "move":
            poi_id = decision.get("target_poi_id")
            if isinstance(poi_id, int):
                # 校验 poi 是否存在且可达
                if any(p.unique_id == poi_id and p.node_id in self.accessible_nodes for p in self.model.poi_agents):
                    return poi_id
                else:
                    self.logger.warning(f"LLM suggested POI {poi_id} not accessible/exists.")
            # 如果 target_poi_id 无效，则回退
            return self._choose_need_based_target()

        # 未识别 action，回退
        self.logger.warning(f"Unknown action from LLM: {decision}")
        return self._choose_need_based_target()
    
    # MOVEMENT DECISION WRAPPERS
    def _make_random_movement_decision(self):
        """
        Make a random movement decision.
        Uses the existing random movement logic but with step-level decision making.
        
        Returns:
            POI ID to move to, 'home' to go home, or None to stay put
        """
        # Simple probability check for random movement
        if random.random() > 0.7:  # 70% chance to stay put for random movement
            if self.model.step_count % 60 == 0:  # Debug every hour
                print(f"DEBUG Random Movement - Resident {self.unique_id}: Staying put")
            return None
        
        # Use existing random target selection
        target = self._choose_random_target()
        if target is None and self.current_node != self.home_node:
            # No suitable POI available and not at home - force going home
            if self.model.step_count % 60 == 0:  # Debug every hour
                print(f"DEBUG Random Movement - Resident {self.unique_id}: No POI available, going home")
            return 'home'
        return target

    def _make_need_based_movement_decision(self):
        """
        Make a need-based movement decision.
        This is where we can later add sophisticated need hierarchy and POI attractiveness.
        
        Returns:
            POI ID to move to, 'home' to go home, or None to stay put
        """
        # Update current needs
        
        
        return None

    def _make_llm_movement_decision(self):
        """Make an LLM-based movement decision."""
        # Removed debug print
        
        # Check if we have a brain
        if not hasattr(self, 'brain') or self.brain is None:
            # Removed debug print
            return self._make_need_based_movement_decision()
        
        # Removed debug print
        target = self._choose_llm_based_target()
        # Removed debug print
        return target

    # =====================================================================
    # 4. NEEDS MANAGEMENT MODULE
    # =====================================================================
    
    def _satisfy_needs_at_poi(self, poi_type):
        """
        Satisfy needs when visiting a POI of a specific type.
        
        Args:
            poi_type: The type of POI being visited
        """
        need_mapping = self.get_need_to_poi_mapping()
        
        # Find which needs this POI type can satisfy
        for need_type, poi_types in need_mapping.items():
            if poi_type in poi_types:
                # Satisfy this need
                satisfaction_amount = random.randint(20, 40)  # Random satisfaction between 20-40
                self.satisfy_need_at_poi(need_type, satisfaction_amount)

    # =====================================================================
    # 6. MEMORY AND STATE MANAGEMENT MODULE
    # =====================================================================
    





    