from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.ollama import OllamaProvider

from .models import CapabilitySummary, SimulationPlan, SimulationRequest, SimulationRunResult
from .simulator import DeepLenseSimulationService


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


@dataclass
class AgentDependencies:
    simulation_service: DeepLenseSimulationService


def build_model_stack(
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    provider: str = "auto",
):
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    ollama = OpenAIChatModel(
        ollama_model,
        provider=OllamaProvider(
            base_url=os.getenv("OLLAMA_BASE_URL", ollama_base_url),
            api_key=os.getenv("OLLAMA_API_KEY"),
        ),
    )

    if provider == "ollama":
        return ollama, f"Ollama only ({ollama_model})"

    if provider == "gemini":
        if not google_api_key:
            raise ValueError(
                "Gemini provider was requested, but GOOGLE_API_KEY/GEMINI_API_KEY is not set."
            )
        google = GoogleModel(
            gemini_model,
            provider=GoogleProvider(api_key=google_api_key),
        )
        return google, f"Gemini only ({gemini_model})"

    if google_api_key:
        google = GoogleModel(
            gemini_model,
            provider=GoogleProvider(api_key=google_api_key),
        )
        return FallbackModel(google, ollama), (
            f"Gemini ({gemini_model}) with Ollama fallback ({ollama_model})"
        )

    return ollama, f"Ollama only ({ollama_model})"


SYSTEM_PROMPT = """
You are DeepLenseSim Agent, a simulation orchestrator for strong gravitational lensing images.

Your job is to turn natural-language requests into validated DeepLenseSim runs.

Current capabilities:
- Supported model configurations: Model_I, Model_II, Model_III.
- Supported substructure types: no_sub, cdm, axion.
- Model_I defaults to 150x150 pixels with a Gaussian PSF.
- Model_II defaults to 64x64 pixels with a Euclid-like instrument.
- Model_III defaults to 64x64 pixels with an HST-like instrument.
- Default lens/source redshifts are 0.5 and 1.0.
- Default main halo mass is 1e12 solar masses.
- Default CDM mean subhalo count is 25.
- Default axion parameters are axion_mass=1e-23 and vortex_mass=3e10 when the user asks for axion structure but omits them.

Conversation rules:
- If the request is missing a crucial choice such as model configuration or substructure type, ask one concise follow-up question.
- If the user asks what is supported, use the capabilities tool.
- Once you have enough information, call the preview tool first and summarize the resolved plan.
- Do not run the simulation immediately after previewing. Ask the user for explicit confirmation first.
- Only call the simulation tool after the user clearly confirms they want to proceed.
- When a run completes, return the structured SimulationRunResult output, not free-form prose.
""".strip()


def build_deeplense_agent(
    simulation_service: DeepLenseSimulationService | None = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    provider: str = "auto",
):
    model, _ = build_model_stack(
        gemini_model=gemini_model,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        provider=provider,
    )

    agent = Agent(
        model=model,
        deps_type=AgentDependencies,
        output_type=[SimulationRunResult, str],
        system_prompt=SYSTEM_PROMPT,
        model_settings={"temperature": 0.1},
    )

    @agent.tool
    def get_supported_configurations(ctx: RunContext[AgentDependencies]) -> CapabilitySummary:
        """Return the supported DeepLenseSim model configurations and substructure options."""

        return ctx.deps.simulation_service.get_capabilities()

    @agent.tool
    def preview_simulation_plan(
        ctx: RunContext[AgentDependencies],
        request: SimulationRequest,
    ) -> SimulationPlan:
        """Resolve defaults, validate the request, and describe the exact simulation plan that would be executed."""

        return ctx.deps.simulation_service.preview(request)

    @agent.tool
    def run_deeplense_simulation(
        ctx: RunContext[AgentDependencies],
        request: SimulationRequest,
    ) -> SimulationRunResult:
        """Execute DeepLenseSim and write image artifacts plus structured metadata to disk."""

        return ctx.deps.simulation_service.run(request)

    return agent
