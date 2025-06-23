import datetime
import json
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Import your existing persona components
import sys
sys.path.append('../../')


@dataclass
class MemoryItem:
    """Enhanced memory item with relevance scoring capabilities."""
    content: str
    timestamp: datetime.datetime
    location: str
    memory_type: str  # 'experience', 'observation', 'reflection', 'conversation'
    importance: float  # 0.0-1.0
    emotional_valence: float  # -1.0 to 1.0 (negative to positive)
    associated_agents: List[str]
    tags: List[str]
    satisfaction_change: float
    needs_addressed: Dict[str, float]
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.recency_score = self._calculate_recency_score()
    
    def _calculate_recency_score(self) -> float:
        """Calculate recency score based on time elapsed."""
        now = datetime.datetime.now()
        hours_elapsed = (now - self.timestamp).total_seconds() / 3600
        # Exponential decay: recent memories more important
        return math.exp(-hours_elapsed / 24.0)  # Half-life of 24 hours


class MemoryRetrievalSystem:
    """Advanced memory retrieval using importance, recency, and relevance."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Retrieval weights (can be personalized per agent)
        self.recency_weight = 0.3
        self.importance_weight = 0.4
        self.relevance_weight = 0.3
        
        # Memory consolidation parameters
        self.reflection_threshold = 150  # Total importance before reflection
        self.max_memories = 500  # Maximum memories to keep
        self.consolidation_interval = 24  # Hours between consolidation
    
    def retrieve_relevant_memories(
        self, 
        memories: List[MemoryItem], 
        query_context: Dict[str, Any],
        top_k: int = 5
    ) -> List[MemoryItem]:
        """
        Retrieve the most relevant memories for a given context.
        
        Args:
            memories: List of all memories
            query_context: Current context (location, needs, agents, etc.)
            top_k: Number of memories to retrieve
            
        Returns:
            List of most relevant memories
        """
        if not memories:
            return []
        
        # Calculate composite scores for all memories
        scored_memories = []
        for memory in memories:
            score = self._calculate_memory_score(memory, query_context)
            scored_memories.append((memory, score))
        
        # Sort by score and return top-k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, score in scored_memories[:top_k]]
    
    def _calculate_memory_score(self, memory: MemoryItem, context: Dict[str, Any]) -> float:
        """Calculate composite relevance score for a memory."""
        
        # Recency score (0.0-1.0)
        recency = memory.recency_score
        
        # Importance score (already 0.0-1.0)
        importance = memory.importance
        
        # Relevance score based on context matching
        relevance = self._calculate_contextual_relevance(memory, context)
        
        # Weighted combination
        composite_score = (
            self.recency_weight * recency +
            self.importance_weight * importance +
            self.relevance_weight * relevance
        )
        
        return composite_score
    
    def _calculate_contextual_relevance(self, memory: MemoryItem, context: Dict[str, Any]) -> float:
        """Calculate how relevant a memory is to the current context."""
        relevance_score = 0.0
        factors_checked = 0
        
        # Location similarity
        if 'location' in context and context['location']:
            if memory.location == context['location']:
                relevance_score += 1.0
            factors_checked += 1
        
        # Agent presence
        if 'nearby_agents' in context and context['nearby_agents']:
            nearby_agent_ids = [str(agent.get('id', '')) for agent in context['nearby_agents']]
            if any(agent_id in memory.associated_agents for agent_id in nearby_agent_ids):
                relevance_score += 0.8
            factors_checked += 1
        
        # Needs similarity
        if 'current_needs' in context and context['current_needs']:
            current_needs = context['current_needs']
            needs_overlap = 0
            for need, current_level in current_needs.items():
                if need in memory.needs_addressed:
                    # Higher relevance if memory addressed a currently high need
                    if current_level > 60:  # High need threshold
                        needs_overlap += memory.needs_addressed[need]
            if needs_overlap > 0:
                relevance_score += min(1.0, needs_overlap / len(current_needs))
            factors_checked += 1
        
        # Tag matching
        if 'query_tags' in context and context['query_tags']:
            tag_matches = len(set(memory.tags) & set(context['query_tags']))
            if tag_matches > 0:
                relevance_score += min(1.0, tag_matches / len(context['query_tags']))
            factors_checked += 1
        
        # Time of day similarity (for routine activities)
        if 'current_time' in context and context['current_time']:
            current_hour = context['current_time'].hour
            memory_hour = memory.timestamp.hour
            hour_diff = min(abs(current_hour - memory_hour), 24 - abs(current_hour - memory_hour))
            if hour_diff <= 2:  # Within 2 hours
                relevance_score += 0.5
            factors_checked += 1
        
        # Normalize by number of factors checked
        return relevance_score / max(1, factors_checked)
    
    def consolidate_memories(self, memories: List[MemoryItem]) -> List[MemoryItem]:
        """
        Consolidate memories by removing less important ones and creating reflections.
        
        Args:
            memories: Current memory list
            
        Returns:
            Consolidated memory list
        """
        if len(memories) <= self.max_memories:
            return memories
        
        # Sort by composite importance (recency + importance + recent access)
        now = datetime.datetime.now()
        scored_memories = []
        
        for memory in memories:
            # Calculate retention score
            age_hours = (now - memory.timestamp).total_seconds() / 3600
            recency_factor = math.exp(-age_hours / 48.0)  # 48-hour half-life for consolidation
            retention_score = memory.importance * 0.7 + recency_factor * 0.3
            scored_memories.append((memory, retention_score))
        
        # Keep top memories
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        consolidated = [memory for memory, score in scored_memories[:self.max_memories]]
        
        self.logger.info(f"Consolidated {len(memories)} memories to {len(consolidated)}")
        return consolidated
    
    def generate_reflection(self, recent_memories: List[MemoryItem], agent_context: Dict[str, Any]) -> Optional[MemoryItem]:
        """
        Generate a reflective memory based on recent experiences.
        
        Args:
            recent_memories: Recent memories to reflect upon
            agent_context: Current agent context
            
        Returns:
            Reflection memory item or None
        """
        if len(recent_memories) < 3:
            return None
        
        # Analyze patterns in recent memories
        locations = [m.location for m in recent_memories]
        emotions = [m.emotional_valence for m in recent_memories]
        satisfactions = [m.satisfaction_change for m in recent_memories]
        
        # Generate reflection content
        avg_satisfaction = sum(satisfactions) / len(satisfactions)
        avg_emotion = sum(emotions) / len(emotions)
        most_common_location = max(set(locations), key=locations.count)
        
        reflection_content = f"Reflecting on recent experiences: "
        if avg_satisfaction > 0.2:
            reflection_content += "Recent activities have been quite satisfying, "
        elif avg_satisfaction < -0.2:
            reflection_content += "Recent activities have been disappointing, "
        else:
            reflection_content += "Recent activities have been mixed, "
        
        if avg_emotion > 0.3:
            reflection_content += "with generally positive feelings. "
        elif avg_emotion < -0.3:
            reflection_content += "with some negative experiences. "
        else:
            reflection_content += "with neutral emotional impact. "
        
        reflection_content += f"Spent most time at {most_common_location}."
        
        # Create reflection memory
        reflection = MemoryItem(
            content=reflection_content,
            timestamp=datetime.datetime.now(),
            location=most_common_location,
            memory_type="reflection",
            importance=0.7,  # Reflections are important
            emotional_valence=avg_emotion,
            associated_agents=[],
            tags=["reflection", "pattern_analysis"],
            satisfaction_change=0.1,  # Small satisfaction from self-understanding
            needs_addressed={}
        )
        
        return reflection


class EnhancedShortTermMemory:
    """
    Enhanced version of ShortTermMemory with better retrieval and integration.
    Combines the structure of the original with advanced memory management.
    """
    
    def __init__(self):
        # Original ShortTermMemory parameters (preserved for compatibility)
        self.vision_r = 4
        self.att_bandwidth = 3
        self.retention = 5
        
        # Time and location
        self.curr_time = None
        self.curr_tile = None
        self.daily_plan_req = None
        
        # Identity (preserved from original)
        self.name = None
        self.first_name = None
        self.last_name = None
        self.age = None
        self.innate = None
        self.learned = None
        self.currently = None
        self.lifestyle = None
        self.living_area = None
        
        # Enhanced memory system
        self.memories: List[MemoryItem] = []
        self.memory_retrieval = MemoryRetrievalSystem()
        self.last_consolidation = datetime.datetime.now()
        
        # Planning (preserved from original)
        self.daily_req = []
        self.f_daily_schedule = []
        self.f_daily_schedule_hourly_org = []
        
        # Current action (preserved from original)
        self.act_address = None
        self.act_start_time = None
        self.act_duration = None
        self.act_description = None
        self.act_pronunciatio = None
        self.act_event = (self.name, None, None)
        
        # Social interaction (preserved from original)
        self.chatting_with = None
        self.chat = None
        self.chatting_with_buffer = dict()
        self.chatting_end_time = None
        
        # Path planning (preserved from original)
        self.act_path_set = False
        self.planned_path = []
        
        # Enhanced reflection parameters
        self.reflection_threshold = 150
        self.importance_accumulator = 0
        
        self.logger = logging.getLogger(f"EnhancedMemory-{self.name}")
    
    def add_memory(self, content: str, memory_type: str, importance: float = 0.5,
                   emotional_valence: float = 0.0, location: str = None,
                   associated_agents: List[str] = None, tags: List[str] = None,
                   satisfaction_change: float = 0.0, needs_addressed: Dict[str, float] = None):
        """Add a new memory with enhanced metadata."""
        
        if location is None:
            location = self.act_address or "unknown"
        if associated_agents is None:
            associated_agents = []
        if tags is None:
            tags = []
        if needs_addressed is None:
            needs_addressed = {}
        
        memory = MemoryItem(
            content=content,
            timestamp=datetime.datetime.now(),
            location=location,
            memory_type=memory_type,
            importance=importance,
            emotional_valence=emotional_valence,
            associated_agents=associated_agents,
            tags=tags,
            satisfaction_change=satisfaction_change,
            needs_addressed=needs_addressed
        )
        
        self.memories.append(memory)
        self.importance_accumulator += importance
        
        # Check if reflection is needed
        if self.importance_accumulator >= self.reflection_threshold:
            self._trigger_reflection()
        
        # Periodic consolidation
        if (datetime.datetime.now() - self.last_consolidation).total_seconds() > 3600:  # Every hour
            self._consolidate_memories()
    
    def retrieve_memories(self, context: Dict[str, Any], top_k: int = 5) -> List[MemoryItem]:
        """Retrieve relevant memories for current context."""
        return self.memory_retrieval.retrieve_relevant_memories(
            self.memories, context, top_k
        )
    
    def _trigger_reflection(self):
        """Trigger a reflection process based on recent experiences."""
        recent_memories = [m for m in self.memories if m.importance > 0.3][-10:]  # Last 10 important memories
        
        reflection = self.memory_retrieval.generate_reflection(
            recent_memories, 
            {"name": self.name, "location": self.act_address}
        )
        
        if reflection:
            self.memories.append(reflection)
            self.logger.info(f"Generated reflection: {reflection.content[:50]}...")
        
        self.importance_accumulator = 0  # Reset accumulator
    
    def _consolidate_memories(self):
        """Consolidate memories to manage memory load."""
        original_count = len(self.memories)
        self.memories = self.memory_retrieval.consolidate_memories(self.memories)
        self.last_consolidation = datetime.datetime.now()
        
        if len(self.memories) < original_count:
            self.logger.info(f"Consolidated memories: {original_count} -> {len(self.memories)}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of current memory state."""
        if not self.memories:
            return {"total_memories": 0, "recent_memories": 0, "important_memories": 0}
        
        now = datetime.datetime.now()
        recent_count = len([m for m in self.memories 
                           if (now - m.timestamp).total_seconds() < 3600])  # Last hour
        important_count = len([m for m in self.memories if m.importance > 0.7])
        
        avg_importance = sum(m.importance for m in self.memories) / len(self.memories)
        avg_emotion = sum(m.emotional_valence for m in self.memories) / len(self.memories)
        
        return {
            "total_memories": len(self.memories),
            "recent_memories": recent_count,
            "important_memories": important_count,
            "avg_importance": round(avg_importance, 2),
            "avg_emotional_valence": round(avg_emotion, 2),
            "importance_accumulator": round(self.importance_accumulator, 2)
        }
    
    # Preserve original interface methods for compatibility
    def get_str_iss(self):
        """Original ISS method preserved for compatibility."""
        commonset = ""
        commonset += f"Name: {self.name}\n"
        commonset += f"Age: {self.age}\n"
        commonset += f"Innate traits: {self.innate}\n"
        commonset += f"Learned traits: {self.learned}\n"
        commonset += f"Currently: {self.currently}\n"
        commonset += f"Lifestyle: {self.lifestyle}\n"
        commonset += f"Daily plan requirement: {self.daily_plan_req}\n"
        if self.curr_time:
            commonset += f"Current Date: {self.curr_time.strftime('%A %B %d')}\n"
        
        # Add memory context
        memory_summary = self.get_memory_summary()
        commonset += f"Memory state: {memory_summary['total_memories']} memories, "
        commonset += f"recent satisfaction: {memory_summary.get('avg_emotional_valence', 0):.1f}\n"
        
        return commonset
    
    def add_new_action(self, action_address, action_duration, action_description,
                       action_pronunciatio, action_event, chatting_with, chat,
                       chatting_with_buffer, chatting_end_time, act_obj_description,
                       act_obj_pronunciatio, act_obj_event, act_start_time=None):
        """Original method preserved, enhanced with memory recording."""
        
        # Store previous action as memory if it existed
        if self.act_description and self.act_start_time:
            # Calculate satisfaction based on action completion
            satisfaction = 0.3  # Default satisfaction for completing an action
            importance = 0.4    # Default importance
            
            # Add action completion memory
            self.add_memory(
                content=f"Completed action: {self.act_description}",
                memory_type="experience",
                importance=importance,
                satisfaction_change=satisfaction,
                location=self.act_address,
                tags=["action_completion", action_description.split()[0] if action_description else "unknown"]
            )
        
        # Set new action (original logic)
        self.act_address = action_address
        self.act_duration = action_duration
        self.act_description = action_description
        self.act_pronunciatio = action_pronunciatio
        self.act_event = action_event
        
        self.chatting_with = chatting_with
        self.chat = chat
        if chatting_with_buffer:
            self.chatting_with_buffer.update(chatting_with_buffer)
        self.chatting_end_time = chatting_end_time
        
        self.act_start_time = self.curr_time
        self.act_path_set = False 