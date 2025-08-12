from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import random

@dataclass
class Action:
    """Simple action representation."""
    name: str                    # e.g., "eat", "sleep", "work"
    location_type: str          # e.g., "home", "restaurant", "workplace", "shop"
    duration_minutes: int       # How long the action takes
    cost: float                # Financial cost of the action
    energy_change: float        # How it affects energy (-1 to 1)
    description: str           # Human-readable description
    
    # Runtime fields
    target_poi_id: Optional[int] = None  # Specific POI to go to
    start_time: Optional[datetime] = None
    remaining_minutes: Optional[int] = None

# Define available everyday actions
EVERYDAY_ACTIONS = {
    "sleep": Action(
        name="sleep",
        location_type="home",
        duration_minutes=480,  # 8 hours
        cost=0.0,
        energy_change=1.0,  # Fully restores energy
        description="Sleep at home to restore energy"
    ),
    "eat_breakfast": Action(
        name="eat_breakfast",
        location_type="home",
        duration_minutes=30,
        cost=5.0,
        energy_change=0.2,
        description="Have breakfast at home"
    ),
    "eat_lunch": Action(
        name="eat_lunch",
        location_type="restaurant",
        duration_minutes=60,
        cost=15.0,
        energy_change=0.3,
        description="Have lunch at a restaurant"
    ),
    "eat_dinner": Action(
        name="eat_dinner",
        location_type="restaurant",
        duration_minutes=90,
        cost=25.0,
        energy_change=0.3,
        description="Have dinner at a restaurant"
    ),
    "buy_groceries": Action(
        name="buy_groceries",
        location_type="supermarket",
        duration_minutes=45,
        cost=50.0,
        energy_change=-0.1,
        description="Shop for groceries"
    ),
    "shower": Action(
        name="shower",
        location_type="home",
        duration_minutes=15,
        cost=0.5,
        energy_change=0.1,
        description="Take a shower at home"
    ),
    "rest": Action(
        name="rest",
        location_type="home",
        duration_minutes=30,
        cost=0.0,
        energy_change=0.2,
        description="Rest and relax at home"
    ),
    "exercise": Action(
        name="exercise",
        location_type="park",
        duration_minutes=60,
        cost=0.0,
        energy_change=-0.3,
        description="Exercise at the park"
    ),
    "socialize": Action(
        name="socialize",
        location_type="cafe",
        duration_minutes=90,
        cost=10.0,
        energy_change=-0.1,
        description="Meet friends at a cafe"
    ),
    "entertainment": Action(
        name="entertainment",
        location_type="entertainment",
        duration_minutes=120,
        cost=20.0,
        energy_change=-0.1,
        description="Enjoy entertainment activities"
    ),
    "work": Action(
        name="work",
        location_type="workplace",
        duration_minutes=480,  # 8 hours
        cost=-200.0,  # Negative cost = earning money
        energy_change=-0.5,
        description="Work at the office"
    )
}

def get_available_actions(hour: int, is_employed: bool, energy: float, money: float) -> Dict[str, Action]:
    """
    Get actions available based on current context.
    
    Args:
        hour: Current hour (0-23)
        is_employed: Whether agent is employed
        energy: Current energy level (0-1)
        money: Current money available
        
    Returns:
        Dictionary of available actions
    """
    available = {}
    
    # Work is only available 9am-5pm for employed agents
    if is_employed and 9 <= hour < 17:
        available["work"] = EVERYDAY_ACTIONS["work"]
    
    # Sleep typically at night or when very tired
    if hour >= 22 or hour < 6 or energy < 0.2:
        available["sleep"] = EVERYDAY_ACTIONS["sleep"]
    
    # Meals at appropriate times
    if 6 <= hour < 10:
        available["eat_breakfast"] = EVERYDAY_ACTIONS["eat_breakfast"]
    if 11 <= hour < 14:
        available["eat_lunch"] = EVERYDAY_ACTIONS["eat_lunch"]
    if 18 <= hour < 21:
        available["eat_dinner"] = EVERYDAY_ACTIONS["eat_dinner"]
    
    # Shopping during business hours
    if 8 <= hour < 20 and money > 50:
        available["buy_groceries"] = EVERYDAY_ACTIONS["buy_groceries"]
    
    # Exercise in morning or evening
    if (6 <= hour < 9 or 17 <= hour < 20) and energy > 0.3:
        available["exercise"] = EVERYDAY_ACTIONS["exercise"]
    
    # Social and entertainment when not working
    if 10 <= hour < 22 and money > 20:
        available["socialize"] = EVERYDAY_ACTIONS["socialize"]
        available["entertainment"] = EVERYDAY_ACTIONS["entertainment"]
    
    # Basic actions always available (with some constraints)
    if energy > 0.1:
        available["shower"] = EVERYDAY_ACTIONS["shower"]
    available["rest"] = EVERYDAY_ACTIONS["rest"]
    
    return available

class ActionMemory:
    """Simple memory system for storing past actions."""
    
    def __init__(self, max_memory_size: int = 20):
        self.max_memory_size = max_memory_size
        self.actions = []  # List of completed actions
        
    def add_action(self, action: Action, completion_time: datetime):
        """Store a completed action."""
        memory_entry = {
            "action_name": action.name,
            "location_type": action.location_type,
            "duration": action.duration_minutes,
            "cost": action.cost,
            "energy_change": action.energy_change,
            "completed_at": completion_time
        }
        
        self.actions.append(memory_entry)
        
        # Keep only recent memories
        if len(self.actions) > self.max_memory_size:
            self.actions.pop(0)
    
    def get_recent_actions(self, n: int = 5) -> list:
        """Get the n most recent actions."""
        return self.actions[-n:] if self.actions else []
    
    def get_summary(self) -> str:
        """Get a text summary of recent actions."""
        if not self.actions:
            return "No recent actions"
        
        recent = self.get_recent_actions(5)
        summary = "Recent actions: "
        for action in recent:
            summary += f"{action['action_name']} (cost: ${action['cost']:.1f}), "
        return summary.rstrip(", ") 