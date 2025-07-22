#!/usr/bin/env python3


import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class PersonaType(Enum):
    """Types of agent personas in healthcare policy simulation."""
    ELDERLY_RESIDENT = "elderly_resident"
    WORKING_PARENT = "working_parent"
    YOUNG_PROFESSIONAL = "young_professional"
    STUDENT = "student"
    CHRONIC_PATIENT = "chronic_patient"
    HEALTHCARE_WORKER = "healthcare_worker"
    POLICY_MAKER = "policy_maker"
    UNEMPLOYED = "unemployed"
    COMMUNITY_LEADER = "community_leader"


class EmotionalState(Enum):
    """Primary emotional states that can influence decision-making."""
    CALM = "calm"
    ANXIOUS = "anxious"
    FRUSTRATED = "frustrated"
    CONFIDENT = "confident"
    STRESSED = "stressed"
    HOPEFUL = "hopeful"
    WORRIED = "worried"
    SATISFIED = "satisfied"


class MotivationType(Enum):
    """Core motivational drivers for agent behavior."""
    HEALTH_SECURITY = "health_security"
    FAMILY_WELLBEING = "family_wellbeing"
    CAREER_ADVANCEMENT = "career_advancement"
    SOCIAL_APPROVAL = "social_approval"
    FINANCIAL_STABILITY = "financial_stability"
    COMMUNITY_SERVICE = "community_service"
    PERSONAL_AUTONOMY = "personal_autonomy"
    KNOWLEDGE_SEEKING = "knowledge_seeking"


@dataclass
class PersonaTemplate:
    """Template defining a specific agent archetype."""
    persona_type: PersonaType
    name: str
    description: str
    demographics: Dict[str, Any]
    core_values: List[str]
    beliefs: Dict[str, str]
    behavioral_tendencies: Dict[str, float]  # 0.0-1.0 scales
    decision_patterns: Dict[str, str]
    healthcare_attitudes: Dict[str, Any]
    social_preferences: Dict[str, float]
    risk_tolerance: float  # 0.0 (risk-averse) to 1.0 (risk-seeking)
    trust_levels: Dict[str, float]  # Trust in different institutions/sources
    communication_style: Dict[str, Any]
    goals_priorities: List[Dict[str, Any]]


@dataclass
class KnowledgeBase:
    """Repository of information accessible to specific agent types."""
    knowledge_id: str
    title: str
    content_type: str  # "policy_document", "health_info", "community_data", etc.
    content: str
    relevance_tags: List[str]
    credibility_score: float  # 0.0-1.0
    last_updated: datetime
    access_permissions: List[PersonaType]


@dataclass
class EmotionalMotivationalState:
    """Dynamic emotional and motivational state of an agent."""
    agent_id: str
    current_emotions: Dict[EmotionalState, float]  # Intensity 0.0-1.0
    motivational_drives: Dict[MotivationType, float]  # Strength 0.0-1.0
    stress_level: float  # 0.0-1.0
    confidence_level: float  # 0.0-1.0
    trust_in_healthcare: float  # 0.0-1.0
    satisfaction_with_services: float  # 0.0-1.0
    recent_experiences: List[Dict[str, Any]]  # Recent events affecting state
    adaptation_rate: float  # How quickly the agent adapts to new situations
    last_updated: datetime


class PersonaTemplateManager:
    """Manages and generates diverse agent persona templates."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.templates = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default persona templates for healthcare policy simulation."""
        
        # Elderly Resident Persona
        self.templates[PersonaType.ELDERLY_RESIDENT] = PersonaTemplate(
            persona_type=PersonaType.ELDERLY_RESIDENT,
            name="Elderly Resident",
            description="Senior citizen with health concerns and limited mobility, values stability and accessible healthcare",
            demographics={
                "age_range": (65, 85),
                "income_level": "fixed_pension",
                "education": ["primary", "secondary"],
                "family_status": "retired",
                "mobility": "limited"
            },
            core_values=["health", "family", "stability", "respect", "independence"],
            beliefs={
                "healthcare_access": "Healthcare should be easily accessible and affordable for seniors",
                "technology": "Prefer traditional methods, cautious about new technology",
                "community": "Strong community ties are essential for well-being",
                "government": "Government should provide adequate support for elderly care"
            },
            behavioral_tendencies={
                "routine_preference": 0.9,
                "technology_adoption": 0.2,
                "social_interaction": 0.7,
                "health_consciousness": 0.9,
                "change_resistance": 0.8
            },
            decision_patterns={
                "healthcare": "Prioritizes proximity and familiarity over advanced features",
                "information_seeking": "Relies on trusted sources like family doctor or community",
                "risk_assessment": "Very cautious, prefers proven solutions"
            },
            healthcare_attitudes={
                "preventive_care": 0.8,
                "trust_in_doctors": 0.9,
                "comfort_with_telemedicine": 0.3,
                "medication_compliance": 0.8
            },
            social_preferences={
                "family_involvement": 0.9,
                "peer_influence": 0.7,
                "authority_deference": 0.8
            },
            risk_tolerance=0.2,
            trust_levels={
                "healthcare_professionals": 0.9,
                "government": 0.6,
                "technology": 0.3,
                "family": 0.95
            },
            communication_style={
                "formality": "high",
                "detail_preference": "moderate",
                "channel_preference": "face_to_face"
            },
            goals_priorities=[
                {"goal": "maintain_health", "priority": 0.9},
                {"goal": "stay_independent", "priority": 0.8},
                {"goal": "family_connection", "priority": 0.7}
            ]
        )
        
        # Working Parent Persona
        self.templates[PersonaType.WORKING_PARENT] = PersonaTemplate(
            persona_type=PersonaType.WORKING_PARENT,
            name="Working Parent",
            description="Busy parent balancing career and family responsibilities, values efficiency and family health",
            demographics={
                "age_range": (30, 50),
                "income_level": "middle_class",
                "education": ["secondary", "university"],
                "family_status": "parent",
                "employment": "full_time"
            },
            core_values=["family", "efficiency", "balance", "security", "achievement"],
            beliefs={
                "healthcare_access": "Healthcare should be convenient and fit into busy schedules",
                "technology": "Technology can help manage health and save time",
                "work_life_balance": "Health decisions must consider work and family constraints",
                "prevention": "Preventive care is important but must be practical"
            },
            behavioral_tendencies={
                "time_consciousness": 0.9,
                "technology_adoption": 0.7,
                "multitasking": 0.8,
                "family_prioritization": 0.9,
                "efficiency_seeking": 0.8
            },
            decision_patterns={
                "healthcare": "Balances quality with convenience and time constraints",
                "information_seeking": "Uses multiple sources, values peer recommendations",
                "scheduling": "Prefers flexible, after-hours, or integrated services"
            },
            healthcare_attitudes={
                "preventive_care": 0.7,
                "trust_in_doctors": 0.8,
                "comfort_with_telemedicine": 0.8,
                "family_health_priority": 0.9
            },
            social_preferences={
                "peer_networks": 0.8,
                "online_communities": 0.6,
                "professional_advice": 0.8
            },
            risk_tolerance=0.5,
            trust_levels={
                "healthcare_professionals": 0.8,
                "peer_recommendations": 0.7,
                "online_reviews": 0.6,
                "employer_benefits": 0.7
            },
            communication_style={
                "formality": "moderate",
                "detail_preference": "concise",
                "channel_preference": "digital_and_phone"
            },
            goals_priorities=[
                {"goal": "family_health", "priority": 0.9},
                {"goal": "work_performance", "priority": 0.7},
                {"goal": "time_efficiency", "priority": 0.8}
            ]
        )
        
        # Young Professional Persona
        self.templates[PersonaType.YOUNG_PROFESSIONAL] = PersonaTemplate(
            persona_type=PersonaType.YOUNG_PROFESSIONAL,
            name="Young Professional",
            description="Career-focused individual with good health, values convenience and modern solutions",
            demographics={
                "age_range": (22, 35),
                "income_level": "middle_to_high",
                "education": ["university", "postgraduate"],
                "family_status": "single_or_couple",
                "employment": "professional"
            },
            core_values=["achievement", "innovation", "efficiency", "lifestyle", "growth"],
            beliefs={
                "healthcare_access": "Healthcare should be modern, efficient, and tech-enabled",
                "technology": "Technology improves healthcare access and quality",
                "prevention": "Preventive care and wellness are investments in future success",
                "autonomy": "Should have control over health decisions and data"
            },
            behavioral_tendencies={
                "technology_adoption": 0.9,
                "innovation_openness": 0.8,
                "self_reliance": 0.7,
                "convenience_seeking": 0.8,
                "data_driven": 0.7
            },
            decision_patterns={
                "healthcare": "Values convenience, technology integration, and evidence-based care",
                "information_seeking": "Uses online resources, apps, and peer networks",
                "provider_selection": "Considers ratings, reviews, and digital capabilities"
            },
            healthcare_attitudes={
                "preventive_care": 0.6,
                "trust_in_doctors": 0.7,
                "comfort_with_telemedicine": 0.9,
                "health_tracking": 0.8
            },
            social_preferences={
                "online_communities": 0.8,
                "professional_networks": 0.7,
                "peer_influence": 0.6
            },
            risk_tolerance=0.6,
            trust_levels={
                "healthcare_professionals": 0.7,
                "technology_platforms": 0.8,
                "peer_reviews": 0.7,
                "data_analytics": 0.8
            },
            communication_style={
                "formality": "low",
                "detail_preference": "data_rich",
                "channel_preference": "digital_first"
            },
            goals_priorities=[
                {"goal": "career_success", "priority": 0.8},
                {"goal": "lifestyle_optimization", "priority": 0.7},
                {"goal": "health_tracking", "priority": 0.6}
            ]
        )
        
        # Chronic Patient Persona
        self.templates[PersonaType.CHRONIC_PATIENT] = PersonaTemplate(
            persona_type=PersonaType.CHRONIC_PATIENT,
            name="Chronic Patient",
            description="Individual managing long-term health condition, highly engaged with healthcare system",
            demographics={
                "age_range": (25, 75),
                "income_level": "variable",
                "education": "variable",
                "health_status": "chronic_condition",
                "healthcare_engagement": "high"
            },
            core_values=["health", "quality_of_life", "support", "information", "advocacy"],
            beliefs={
                "healthcare_access": "Continuous, coordinated care is essential for managing chronic conditions",
                "patient_advocacy": "Patients must be active participants in their care",
                "support_systems": "Peer support and family involvement are crucial",
                "research": "Staying informed about treatment advances is important"
            },
            behavioral_tendencies={
                "health_monitoring": 0.9,
                "information_seeking": 0.9,
                "treatment_compliance": 0.8,
                "support_seeking": 0.8,
                "advocacy": 0.7
            },
            decision_patterns={
                "healthcare": "Prioritizes quality, continuity, and specialization",
                "provider_relationships": "Values long-term relationships with trusted providers",
                "treatment_decisions": "Carefully weighs benefits and risks, seeks second opinions"
            },
            healthcare_attitudes={
                "preventive_care": 0.9,
                "trust_in_doctors": 0.8,
                "comfort_with_telemedicine": 0.7,
                "medication_compliance": 0.9
            },
            social_preferences={
                "support_groups": 0.8,
                "patient_communities": 0.9,
                "family_involvement": 0.8
            },
            risk_tolerance=0.4,
            trust_levels={
                "specialists": 0.9,
                "patient_organizations": 0.8,
                "research_institutions": 0.8,
                "peer_experiences": 0.7
            },
            communication_style={
                "formality": "moderate",
                "detail_preference": "comprehensive",
                "channel_preference": "multi_channel"
            },
            goals_priorities=[
                {"goal": "symptom_management", "priority": 0.9},
                {"goal": "quality_of_life", "priority": 0.8},
                {"goal": "treatment_access", "priority": 0.9}
            ]
        )
        
        self.logger.info(f"Initialized {len(self.templates)} persona templates")
    
    def get_template(self, persona_type: PersonaType) -> PersonaTemplate:
        """Get a specific persona template."""
        return self.templates.get(persona_type)
    
    def generate_persona_variant(self, base_persona: PersonaType, variation_factor: float = 0.1) -> PersonaTemplate:
        """Generate a variant of a base persona with some randomization."""
        base_template = self.templates[base_persona]
        
        # Create a copy and add variation
        variant = PersonaTemplate(
            persona_type=base_template.persona_type,
            name=f"{base_template.name}_variant",
            description=base_template.description,
            demographics=base_template.demographics.copy(),
            core_values=base_template.core_values.copy(),
            beliefs=base_template.beliefs.copy(),
            behavioral_tendencies=self._add_variation(base_template.behavioral_tendencies, variation_factor),
            decision_patterns=base_template.decision_patterns.copy(),
            healthcare_attitudes=self._add_variation(base_template.healthcare_attitudes, variation_factor),
            social_preferences=self._add_variation(base_template.social_preferences, variation_factor),
            risk_tolerance=max(0.0, min(1.0, base_template.risk_tolerance + random.uniform(-variation_factor, variation_factor))),
            trust_levels=self._add_variation(base_template.trust_levels, variation_factor),
            communication_style=base_template.communication_style.copy(),
            goals_priorities=base_template.goals_priorities.copy()
        )
        
        return variant
    
    def _add_variation(self, values_dict: Dict[str, float], variation_factor: float) -> Dict[str, float]:
        """Add random variation to a dictionary of float values."""
        varied_dict = {}
        for key, value in values_dict.items():
            variation = random.uniform(-variation_factor, variation_factor)
            varied_dict[key] = max(0.0, min(1.0, value + variation))
        return varied_dict
    
    def get_all_templates(self) -> Dict[PersonaType, PersonaTemplate]:
        """Get all available persona templates."""
        return self.templates.copy()


class KnowledgeBaseManager:
    """Manages knowledge repositories for different agent types using RAG principles."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.knowledge_bases = {}
        self._initialize_default_knowledge_bases()
    
    def _initialize_default_knowledge_bases(self):
        """Initialize default knowledge bases for healthcare policy simulation."""
        
        # Healthcare Policy Knowledge
        self.knowledge_bases["healthcare_policy"] = KnowledgeBase(
            knowledge_id="healthcare_policy",
            title="Healthcare Policy Information",
            content_type="policy_document",
            content="""
            Macau Healthcare System Overview:
            - Universal healthcare coverage through Social Security Fund
            - Public hospitals: Centro Hospitalar Conde de São Januário, Hospital Kiang Wu
            - Primary care through health centers in each parish
            - Specialized services: cardiology, oncology, pediatrics
            - Emergency services available 24/7
            - Prescription medication subsidies for chronic conditions
            - Elderly care programs and home health services
            - Mental health services and counseling
            - Preventive care programs: vaccinations, health screenings
            """,
            relevance_tags=["healthcare", "policy", "access", "services"],
            credibility_score=0.9,
            last_updated=datetime.now(),
            access_permissions=[PersonaType.POLICY_MAKER, PersonaType.HEALTHCARE_WORKER, PersonaType.COMMUNITY_LEADER]
        )
        
        # Community Health Information
        self.knowledge_bases["community_health"] = KnowledgeBase(
            knowledge_id="community_health",
            title="Community Health Resources",
            content_type="health_info",
            content="""
            Taipa and Coloane Health Resources:
            - Taipa Health Center: Family medicine, pediatrics, basic diagnostics
            - Coloane Health Station: Primary care, elderly services
            - Local pharmacies: 24-hour availability, prescription services
            - Community wellness programs: fitness classes, health education
            - Support groups: diabetes management, mental health, elderly care
            - Traditional Chinese Medicine clinics
            - Dental services and optical care
            - Health screening programs for residents
            """,
            relevance_tags=["community", "health", "resources", "local"],
            credibility_score=0.8,
            last_updated=datetime.now(),
            access_permissions=[PersonaType.ELDERLY_RESIDENT, PersonaType.WORKING_PARENT, 
                             PersonaType.CHRONIC_PATIENT, PersonaType.COMMUNITY_LEADER]
        )
        
        # Patient Rights and Advocacy
        self.knowledge_bases["patient_rights"] = KnowledgeBase(
            knowledge_id="patient_rights",
            title="Patient Rights and Advocacy Information",
            content_type="legal_info",
            content="""
            Patient Rights in Macau Healthcare:
            - Right to accessible and quality healthcare
            - Right to informed consent for treatments
            - Right to privacy and confidentiality
            - Right to second opinions and specialist referrals
            - Right to interpret services for non-Chinese speakers
            - Right to file complaints and seek resolution
            - Right to access medical records
            - Right to refuse treatment
            - Right to emergency care regardless of ability to pay
            """,
            relevance_tags=["rights", "advocacy", "legal", "patients"],
            credibility_score=0.95,
            last_updated=datetime.now(),
            access_permissions=[PersonaType.CHRONIC_PATIENT, PersonaType.HEALTHCARE_WORKER, 
                             PersonaType.COMMUNITY_LEADER]
        )
        
        self.logger.info(f"Initialized {len(self.knowledge_bases)} knowledge bases")
    
    def get_relevant_knowledge(self, agent_persona: PersonaType, query_tags: List[str]) -> List[KnowledgeBase]:
        """Retrieve knowledge bases relevant to an agent's persona and query."""
        relevant_knowledge = []
        
        for kb in self.knowledge_bases.values():
            # Check if agent has access permission
            if agent_persona in kb.access_permissions:
                # Check if any query tags match knowledge base tags
                if any(tag in kb.relevance_tags for tag in query_tags):
                    relevant_knowledge.append(kb)
        
        # Sort by credibility score (descending)
        relevant_knowledge.sort(key=lambda x: x.credibility_score, reverse=True)
        return relevant_knowledge
    
    def add_knowledge_base(self, knowledge_base: KnowledgeBase):
        """Add a new knowledge base to the repository."""
        self.knowledge_bases[knowledge_base.knowledge_id] = knowledge_base
        self.logger.info(f"Added knowledge base: {knowledge_base.title}")
    
    def update_knowledge_base(self, knowledge_id: str, new_content: str):
        """Update the content of an existing knowledge base."""
        if knowledge_id in self.knowledge_bases:
            self.knowledge_bases[knowledge_id].content = new_content
            self.knowledge_bases[knowledge_id].last_updated = datetime.now()
            self.logger.info(f"Updated knowledge base: {knowledge_id}")


class EmotionalMotivationalTracker:
    """Tracks and updates dynamic emotional and motivational states of agents."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agent_states = {}
    
    def initialize_agent_state(self, agent_id: str, persona_template: PersonaTemplate) -> EmotionalMotivationalState:
        """Initialize emotional and motivational state for a new agent."""
        
        # Initialize emotions based on persona
        initial_emotions = self._derive_initial_emotions(persona_template)
        
        # Initialize motivational drives
        initial_motivations = self._derive_initial_motivations(persona_template)
        
        # Create the state
        state = EmotionalMotivationalState(
            agent_id=agent_id,
            current_emotions=initial_emotions,
            motivational_drives=initial_motivations,
            stress_level=0.3,  # Baseline stress
            confidence_level=0.6,  # Moderate confidence
            trust_in_healthcare=persona_template.trust_levels.get("healthcare_professionals", 0.7),
            satisfaction_with_services=0.5,  # Neutral starting point
            recent_experiences=[],
            adaptation_rate=0.1,  # How quickly agent adapts
            last_updated=datetime.now()
        )
        
        self.agent_states[agent_id] = state
        self.logger.debug(f"Initialized emotional state for agent {agent_id}")
        return state
    
    def _derive_initial_emotions(self, persona_template: PersonaTemplate) -> Dict[EmotionalState, float]:
        """Derive initial emotional state from persona template."""
        emotions = {}
        
        # Base emotions for all agents
        emotions[EmotionalState.CALM] = 0.5
        emotions[EmotionalState.CONFIDENT] = persona_template.trust_levels.get("healthcare_professionals", 0.6)
        
        # Persona-specific emotional tendencies
        if persona_template.persona_type == PersonaType.ELDERLY_RESIDENT:
            emotions[EmotionalState.WORRIED] = 0.6
            emotions[EmotionalState.ANXIOUS] = 0.4
        elif persona_template.persona_type == PersonaType.WORKING_PARENT:
            emotions[EmotionalState.STRESSED] = 0.5
            emotions[EmotionalState.WORRIED] = 0.4
        elif persona_template.persona_type == PersonaType.CHRONIC_PATIENT:
            emotions[EmotionalState.ANXIOUS] = 0.7
            emotions[EmotionalState.HOPEFUL] = 0.6
        else:
            emotions[EmotionalState.CALM] = 0.6
            emotions[EmotionalState.CONFIDENT] = 0.7
        
        return emotions
    
    def _derive_initial_motivations(self, persona_template: PersonaTemplate) -> Dict[MotivationType, float]:
        """Derive initial motivational drives from persona template."""
        motivations = {}
        
        # Map persona goals to motivational drives
        for goal in persona_template.goals_priorities:
            if "health" in goal["goal"]:
                motivations[MotivationType.HEALTH_SECURITY] = goal["priority"]
            elif "family" in goal["goal"]:
                motivations[MotivationType.FAMILY_WELLBEING] = goal["priority"]
            elif "career" in goal["goal"] or "work" in goal["goal"]:
                motivations[MotivationType.CAREER_ADVANCEMENT] = goal["priority"]
        
        # Add default motivations based on persona type
        if persona_template.persona_type == PersonaType.ELDERLY_RESIDENT:
            motivations[MotivationType.HEALTH_SECURITY] = 0.9
            motivations[MotivationType.PERSONAL_AUTONOMY] = 0.7
        elif persona_template.persona_type == PersonaType.WORKING_PARENT:
            motivations[MotivationType.FAMILY_WELLBEING] = 0.9
            motivations[MotivationType.FINANCIAL_STABILITY] = 0.7
        elif persona_template.persona_type == PersonaType.CHRONIC_PATIENT:
            motivations[MotivationType.HEALTH_SECURITY] = 0.95
            motivations[MotivationType.KNOWLEDGE_SEEKING] = 0.8
        
        return motivations
    
    def update_state_from_experience(self, agent_id: str, experience: Dict[str, Any]):
        """Update agent's emotional and motivational state based on a recent experience."""
        if agent_id not in self.agent_states:
            self.logger.warning(f"No state found for agent {agent_id}")
            return
        
        state = self.agent_states[agent_id]
        
        # Add experience to recent experiences
        state.recent_experiences.append({
            **experience,
            "timestamp": datetime.now()
        })
        
        # Keep only recent experiences (last 10)
        if len(state.recent_experiences) > 10:
            state.recent_experiences = state.recent_experiences[-10:]
        
        # Update emotional state based on experience
        self._update_emotions_from_experience(state, experience)
        
        # Update motivational drives
        self._update_motivations_from_experience(state, experience)
        
        # Update other state variables
        self._update_general_state(state, experience)
        
        state.last_updated = datetime.now()
        self.logger.debug(f"Updated state for agent {agent_id} based on experience: {experience.get('type', 'unknown')}")
    
    def _update_emotions_from_experience(self, state: EmotionalMotivationalState, experience: Dict[str, Any]):
        """Update emotional state based on experience type and outcome."""
        experience_type = experience.get("type", "")
        outcome = experience.get("outcome", "neutral")
        satisfaction = experience.get("satisfaction", 0.5)
        
        adaptation = state.adaptation_rate
        
        if experience_type == "healthcare_visit":
            if outcome == "positive":
                state.current_emotions[EmotionalState.SATISFIED] = min(1.0, 
                    state.current_emotions.get(EmotionalState.SATISFIED, 0.5) + adaptation)
                state.current_emotions[EmotionalState.CONFIDENT] = min(1.0,
                    state.current_emotions.get(EmotionalState.CONFIDENT, 0.5) + adaptation * 0.5)
                state.current_emotions[EmotionalState.ANXIOUS] = max(0.0,
                    state.current_emotions.get(EmotionalState.ANXIOUS, 0.3) - adaptation)
            else:
                state.current_emotions[EmotionalState.FRUSTRATED] = min(1.0,
                    state.current_emotions.get(EmotionalState.FRUSTRATED, 0.2) + adaptation)
                state.current_emotions[EmotionalState.ANXIOUS] = min(1.0,
                    state.current_emotions.get(EmotionalState.ANXIOUS, 0.3) + adaptation * 0.5)
        
        elif experience_type == "waiting_time":
            if satisfaction < 0.3:  # Long wait
                state.current_emotions[EmotionalState.FRUSTRATED] = min(1.0,
                    state.current_emotions.get(EmotionalState.FRUSTRATED, 0.2) + adaptation)
                state.current_emotions[EmotionalState.STRESSED] = min(1.0,
                    state.current_emotions.get(EmotionalState.STRESSED, 0.3) + adaptation * 0.5)
    
    def _update_motivations_from_experience(self, state: EmotionalMotivationalState, experience: Dict[str, Any]):
        """Update motivational drives based on experience."""
        experience_type = experience.get("type", "")
        outcome = experience.get("outcome", "neutral")
        
        adaptation = state.adaptation_rate * 0.5  # Motivations change more slowly
        
        if experience_type == "healthcare_visit" and outcome == "positive":
            # Positive healthcare experience reinforces health security motivation
            state.motivational_drives[MotivationType.HEALTH_SECURITY] = min(1.0,
                state.motivational_drives.get(MotivationType.HEALTH_SECURITY, 0.7) + adaptation)
        
        elif experience_type == "policy_change":
            # Policy changes can affect various motivations
            if "healthcare_access" in experience.get("details", ""):
                state.motivational_drives[MotivationType.HEALTH_SECURITY] = min(1.0,
                    state.motivational_drives.get(MotivationType.HEALTH_SECURITY, 0.7) + adaptation)
    
    def _update_general_state(self, state: EmotionalMotivationalState, experience: Dict[str, Any]):
        """Update general state variables like stress, confidence, trust."""
        satisfaction = experience.get("satisfaction", 0.5)
        experience_type = experience.get("type", "")
        
        adaptation = state.adaptation_rate
        
        # Update satisfaction with services
        if experience_type in ["healthcare_visit", "service_interaction"]:
            state.satisfaction_with_services = (state.satisfaction_with_services * 0.8 + 
                                              satisfaction * 0.2)
        
        # Update trust in healthcare
        if experience_type == "healthcare_visit":
            trust_change = (satisfaction - 0.5) * adaptation
            state.trust_in_healthcare = max(0.0, min(1.0, 
                state.trust_in_healthcare + trust_change))
        
        # Update stress level based on negative experiences
        if satisfaction < 0.3:
            state.stress_level = min(1.0, state.stress_level + adaptation * 0.5)
        elif satisfaction > 0.7:
            state.stress_level = max(0.0, state.stress_level - adaptation * 0.3)
    
    def get_agent_state(self, agent_id: str) -> Optional[EmotionalMotivationalState]:
        """Get the current emotional and motivational state of an agent."""
        return self.agent_states.get(agent_id)
    
    def get_dominant_emotion(self, agent_id: str) -> Optional[EmotionalState]:
        """Get the dominant emotion for an agent."""
        state = self.agent_states.get(agent_id)
        if not state:
            return None
        
        if not state.current_emotions:
            return EmotionalState.CALM
        
        return max(state.current_emotions.items(), key=lambda x: x[1])[0]
    
    def get_primary_motivation(self, agent_id: str) -> Optional[MotivationType]:
        """Get the primary motivational drive for an agent."""
        state = self.agent_states.get(agent_id)
        if not state:
            return None
        
        if not state.motivational_drives:
            return MotivationType.HEALTH_SECURITY
        
        return max(state.motivational_drives.items(), key=lambda x: x[1])[0]


class PersonaMemoryManager:
    """Main interface for managing agent personas and memory systems."""
    
    def __init__(self):
        self.persona_manager = PersonaTemplateManager()
        self.knowledge_manager = KnowledgeBaseManager()
        self.emotional_tracker = EmotionalMotivationalTracker()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Persona Memory Manager initialized")
    
    def create_agent_persona(self, agent_id: str, persona_type: PersonaType, 
                           variation_factor: float = 0.1) -> Tuple[PersonaTemplate, EmotionalMotivationalState]:
        """Create a complete persona profile for an agent."""
        
        # Get base template and create variant
        if variation_factor > 0:
            persona_template = self.persona_manager.generate_persona_variant(persona_type, variation_factor)
        else:
            persona_template = self.persona_manager.get_template(persona_type)
        
        # Initialize emotional and motivational state
        emotional_state = self.emotional_tracker.initialize_agent_state(agent_id, persona_template)
        
        self.logger.info(f"Created persona for agent {agent_id}: {persona_type.value}")
        return persona_template, emotional_state
    
    def get_contextual_knowledge(self, agent_id: str, persona_type: PersonaType, 
                               query_context: List[str]) -> str:
        """Get relevant knowledge for an agent's decision-making context."""
        
        # Retrieve relevant knowledge bases
        relevant_knowledge = self.knowledge_manager.get_relevant_knowledge(persona_type, query_context)
        
        if not relevant_knowledge:
            return "No specific knowledge available for this context."
        
        # Combine knowledge from multiple sources
        knowledge_text = "Relevant Information:\n"
        for kb in relevant_knowledge[:3]:  # Limit to top 3 most relevant
            knowledge_text += f"\n{kb.title}:\n{kb.content}\n"
        
        return knowledge_text
    
    def update_agent_experience(self, agent_id: str, experience: Dict[str, Any]):
        """Update an agent's state based on a new experience."""
        self.emotional_tracker.update_state_from_experience(agent_id, experience)
    
    def get_agent_context_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get a comprehensive context summary for an agent."""
        emotional_state = self.emotional_tracker.get_agent_state(agent_id)
        
        if not emotional_state:
            return {"error": "Agent state not found"}
        
        dominant_emotion = self.emotional_tracker.get_dominant_emotion(agent_id)
        primary_motivation = self.emotional_tracker.get_primary_motivation(agent_id)
        
        return {
            "agent_id": agent_id,
            "dominant_emotion": dominant_emotion.value if dominant_emotion else "unknown",
            "primary_motivation": primary_motivation.value if primary_motivation else "unknown",
            "stress_level": emotional_state.stress_level,
            "confidence_level": emotional_state.confidence_level,
            "trust_in_healthcare": emotional_state.trust_in_healthcare,
            "satisfaction_with_services": emotional_state.satisfaction_with_services,
            "recent_experiences_count": len(emotional_state.recent_experiences)
        }
    
    def get_available_personas(self) -> List[PersonaType]:
        """Get list of available persona types."""
        return list(self.persona_manager.get_all_templates().keys()) 