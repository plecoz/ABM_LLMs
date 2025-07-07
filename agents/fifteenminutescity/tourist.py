from __future__ import annotations

import random
from typing import List, Optional

from agents.base_person_agent import BaseAgent
from brains.concordia_brain import ConcordiaBrain

# We only need basic shapely geometry for location representation
from shapely.geometry.base import BaseGeometry


class Tourist(BaseAgent):
    """A simple tourist agent that picks attractions using a Concordia LLM brain."""

    def __init__(
        self,
        model,
        unique_id: int,
        geometry: BaseGeometry,
        itinerary: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, unique_id, geometry, **kwargs)
        self.brain = ConcordiaBrain(name=f"Tourist-{unique_id}")
        self.itinerary = itinerary or []  # simple list of POI types/names to visit
        self.current_target: Optional[str] = None

    # ------------------------------------------------------------------
    # Mesa step
    # ------------------------------------------------------------------
    def step(self):
        # Standard bookkeeping from BaseAgent (location history etc.)
        super().step()

        # If already at destination (or none), decide next move via LLM
        if not self.current_target:
            self.current_target = self._decide_next_destination()

        # Here we would normally trigger movement towards `self.current_target`.
        # For brevity, we simply pop it off the itinerary to mark it as visited.
        if self.current_target and self.current_target in self.itinerary:
            self.itinerary.remove(self.current_target)
        # Reset for next decision
        self.current_target = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _decide_next_destination(self) -> Optional[str]:
        remaining = ", ".join(self.itinerary) if self.itinerary else "none"
        observation = (
            f"You are a tourist. Attractions left: {remaining}. "
            f"Current position: {self.geometry}."
        )
        self.brain.observe(observation)
        try:
            reply = self.brain.decide(
                "Which attraction will you visit next? Reply with the name or 'stay'."
            ).lower()
        except Exception:
            return None

        if 'stay' in reply or not reply:
            return None
        return reply.strip()