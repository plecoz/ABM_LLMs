from __future__ import annotations

"""Generic Concordia LLM brain usable by any Mesa agent.

This file bundles:
• ConcordiaBrain – high-level wrapper exposing `observe()` / `decide()`.
• Light-weight ContextComponents: RecentObservation, SimpleMemory, PersonaContext.
• CustomOpenAILanguageModel – OpenAI-compatible wrapper that allows custom `base_url`.

Place this file at `brains/concordia_brain.py` so that *any* scenario (fifteen-minutes-city or others)
can simply

    from brains.concordia_brain import ConcordiaBrain

and get an out-of-the-box LLM brain with memory + persona.
"""

import sys
import pathlib
from importlib import import_module
from typing import List, Optional

# -----------------------------------------------------------------------------
# Ensure Concordia repo on path
# -----------------------------------------------------------------------------
_repo_path = pathlib.Path(__file__).resolve().parents[1] / "ref_Concordia_repo"
if _repo_path.exists() and str(_repo_path) not in sys.path:
    sys.path.append(str(_repo_path))

# pylint: disable=wrong-import-position
from concordia.agents.entity_agent import EntityAgent
from concordia.components.agent.concat_act_component import ConcatActComponent
from concordia.language_model.no_language_model import NoLanguageModel
from concordia.language_model.gpt_model import GptLanguageModel
from concordia.typing import entity as concordia_entity

# ---------------------- 官方上下文组件 ----------------------
from concordia.components.agent.memory import (
    ListMemory,
    DEFAULT_MEMORY_COMPONENT_KEY as _MEM_KEY,
)
from concordia.components.agent.observation import ObservationToMemory, LastNObservations

# -----------------------------------------------------------------------------
# Custom OpenAI wrapper with base_url support （保留）
# -----------------------------------------------------------------------------
import openai  # noqa: E402
from concordia.language_model.base_gpt_model import BaseGPTModel  # noqa: E402
from concordia.utils.deprecated import measurements as measurements_lib  # noqa: E402


class CustomOpenAILanguageModel(BaseGPTModel):
    """OpenAI-compatible language model allowing custom endpoint."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        measurements: measurements_lib.Measurements | None = None,
        channel: str = "stats.custom_openai",
    ) -> None:
        client = (
            openai.OpenAI(api_key=api_key, base_url=base_url)
            if base_url else openai.OpenAI(api_key=api_key)
        )
        super().__init__(model_name=model_name, client=client, measurements=measurements, channel=channel)


# -----------------------------------------------------------------------------
# Context Components
# -----------------------------------------------------------------------------
from concordia.typing import entity_component  # noqa: E402


# 移除自定义 RecentObservation / SimpleMemory，改用官方组件


class PersonaContext(entity_component.ContextComponent):
    """Static persona description."""

    def __init__(self, persona, label: str = "Persona"):
        super().__init__()
        self._persona = str(persona).strip()
        self._label = label

    def pre_act(self, _):
        return f"{self._label}:\n{self._persona}\n" if self._persona else ""

    def pre_observe(self, _):
        return ""

    def update(self):
        pass

    def get_state(self):
        """Return the current state of the persona context."""
        return {
            'persona': self._persona,
            'label': self._label
        }

    def set_state(self, state):
        """Set the state of the persona context."""
        if isinstance(state, dict):
            self._persona = str(state.get('persona', '')).strip()
            self._label = state.get('label', 'Persona')
        else:
            # Fallback for simple string input
            self._persona = str(state).strip()


# -----------------------------------------------------------------------------
# ConcordiaBrain
# -----------------------------------------------------------------------------

# Load global config if present
try:
    llm_conf = import_module("config.llm_config")
except ModuleNotFoundError:
    llm_conf = None

_PROVIDER_MAP = {
    "openai": GptLanguageModel,
    "custom_openai": CustomOpenAILanguageModel,
}


class ConcordiaBrain:
    """High-level brain that plugs Concordia into any Mesa Agent."""

    def __init__(self, name: str, *, persona: str = "", language_model=None):
        self._model = language_model or self._model_from_config()

        act_component = ConcatActComponent(model=self._model, prefix_entity_name=False)

        # -------------------- 组装官方组件 --------------------
        ctx_components = {
            _MEM_KEY: ListMemory(memory_bank=[]),         # 核心记忆存储
            "obs_writer": ObservationToMemory(),          # 把 observation 写入记忆
            "obs": LastNObservations(history_length=10), # 决策时回显最近观察
        }

        # 若传入 persona，则追加 persona context（沿用本文件定义的PersonaContext）
        if persona:
            ctx_components["persona"] = PersonaContext(persona)

        self.agent = EntityAgent(
            agent_name=name,
            act_component=act_component,
            context_processor=None,
            context_components=ctx_components,
        )

    # ---------------- public ----------------
    def observe(self, text: str):
        # Log observations at specific steps for debugging
        if hasattr(self, '_step_counter'):
            # Use the same counter as decide() method
            pass
        else:
            self._step_counter = 0
            
        # Log observations at steps 1 and 10 (only for action system, not persona test)
        if hasattr(self, '_step_counter') and self._step_counter in [0, 9]:  # 0-based for observations (before decision)
            print(f"\n🔍 OBSERVATION DEBUG - {self.agent.name} - Step {self._step_counter + 1}")
            print("=" * 80)
            print(f"OBSERVATION SENT TO BRAIN:")
            print(text)
            print("=" * 80)
        
        self.agent.observe(text)

    def decide(self, prompt: str) -> str:
        # Log prompts at specific steps for debugging
        if hasattr(self, '_step_counter'):
            self._step_counter += 1
        else:
            self._step_counter = 1
            
        # Log prompts at steps 1 and 10
        if self._step_counter in [1, 10]:
            print(f"\n🔍 DECISION DEBUG - {self.agent.name} - Step {self._step_counter}")
            print("=" * 80)
            print(f"DECISION PROMPT:")
            print(prompt)
            print("=" * 80)
            
            # Show what we can about the brain's state
            try:
                contexts = self.agent.get_all_context_components()
                print(f" BRAIN STATE - {self.agent.name} - Step {self._step_counter}")
                print("=" * 80)
                print("BRAIN COMPONENTS:")
                
                # Show available components
                for name, component in contexts.items():
                    component_type = type(component).__name__
                    print(f"- [{name.upper()}]: {component_type}")
                    
                    # Try to get some info about the component
                    if hasattr(component, '_memory_bank') and hasattr(component._memory_bank, '__len__'):
                        print(f"  └─ Memory entries: {len(component._memory_bank)}")
                    elif hasattr(component, '_persona'):
                        persona_preview = str(component._persona)[:100] + "..." if len(str(component._persona)) > 100 else str(component._persona)
                        print(f"  └─ Persona: {persona_preview}")
                    elif hasattr(component, '_history') and hasattr(component._history, '__len__'):
                        print(f"  └─ History entries: {len(component._history)}")
                
                print("\n" + "-" * 40)
                print(f"DECISION PROMPT: {prompt}")
                print("=" * 80)
                
            except Exception as e:
                print(f"🔍 BRAIN STATE - {self.agent.name} - Step {self._step_counter}")
                print("=" * 80)
                print(f"Error getting brain state: {e}")
                print(f"DECISION PROMPT: {prompt}")
                print("=" * 80)
        
        spec = concordia_entity.ActionSpec(call_to_action=prompt, output_type=concordia_entity.OutputType.FREE)
        response = self.agent.act(spec)
        
        # Also log the response at steps 1 and 10
        if self._step_counter in [1, 10]:
            print(f"🔍 RESPONSE DEBUG - {self.agent.name} - Step {self._step_counter}")
            print("=" * 80)
            print(f"LLM RESPONSE:")
            print(response)
            print("=" * 80)
        
        return response.strip() if response else ""

    def set_persona(self, persona_text: str):
        persona_text = str(persona_text)
        # Access internal dict directly (EntityAgent stores a real dict internally)
        if "persona" in self.agent.get_all_context_components():
            comp = self.agent.get_component("persona", type_=PersonaContext)
            comp._persona = persona_text  # type: ignore  # pylint: disable=protected-access
        else:
            self.agent._context_components["persona"] = PersonaContext(persona_text)  # type: ignore[attr-defined, protected-access]

    # ---------------- internal ----------------
    def _model_from_config(self):
        if llm_conf is None:
            return NoLanguageModel()

        provider = getattr(llm_conf, "PROVIDER", "openai").lower()
        model_name = getattr(llm_conf, "MODEL_NAME", "gpt-4o-mini")
        api_key = getattr(llm_conf, "API_KEY", "") or None
        base_url = getattr(llm_conf, "BASE_URL", None)

        ModelCls = _PROVIDER_MAP.get(provider, NoLanguageModel)
        if ModelCls is NoLanguageModel:
            return NoLanguageModel()

        try:
            if provider == "custom_openai":
                return ModelCls(model_name=model_name, api_key=api_key, base_url=base_url)
            else:  # openai or others
                return ModelCls(model_name=model_name, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            print(f"[ConcordiaBrain] LLM init failed ({provider}): {exc}. Switching to NoLanguageModel.")
            return NoLanguageModel() 