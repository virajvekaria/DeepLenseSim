from __future__ import annotations

import argparse
import sys

from pydantic_ai.messages import ModelRequest, ToolReturnPart

from .agent import (
    AgentDependencies,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    build_deeplense_agent,
    build_model_stack,
)
from .models import SimulationRunResult
from .simulator import DeepLenseSimulationService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Natural-language DeepLenseSim agent built with Pydantic AI."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Optional initial prompt. If omitted, the agent starts an interactive session.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/deeplense_agent",
        help="Directory under which run artifacts will be written.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "ollama", "gemini"],
        default="auto",
        help="Provider selection. Use 'ollama' to force local inference even if a Gemini key is set.",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model name to use when GOOGLE_API_KEY or GEMINI_API_KEY is set.",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name used as the local fallback.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="OpenAI-compatible Ollama base URL.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final SimulationRunResult objects as formatted JSON.",
    )
    return parser


def extract_tool_result(result) -> SimulationRunResult | None:
    for message in reversed(result.all_messages()):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, ToolReturnPart) and part.tool_name == "run_deeplense_simulation":
                if isinstance(part.content, SimulationRunResult):
                    return part.content
    return None


def print_result(result: SimulationRunResult, as_json: bool) -> None:
    if as_json:
        print(result.model_dump_json(indent=2))
        return

    print("simulation complete")
    print(f"run_id: {result.run_id}")
    print(f"output_dir: {result.output_dir}")
    print(f"metadata_path: {result.metadata_path}")
    if result.contact_sheet_path:
        print(f"contact_sheet_path: {result.contact_sheet_path}")
    for artifact in result.artifacts:
        print(
            f"image_{artifact.index:03d}: png={artifact.png_path} npy={artifact.npy_path} "
            f"shape={artifact.shape} min={artifact.min_value:.4g} max={artifact.max_value:.4g}"
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    simulation_service = DeepLenseSimulationService(output_root=args.output_root)
    agent = build_deeplense_agent(
        simulation_service=simulation_service,
        gemini_model=args.gemini_model,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        provider=args.provider,
    )
    deps = AgentDependencies(simulation_service=simulation_service)
    _, stack_summary = build_model_stack(
        gemini_model=args.gemini_model,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        provider=args.provider,
    )

    print(f"model_stack: {stack_summary}")
    print("type 'exit' or 'quit' to end the session.")

    history = None
    pending_prompt = args.prompt
    interactive = pending_prompt is None or sys.stdin.isatty()

    while True:
        if pending_prompt is not None:
            user_prompt = pending_prompt
            pending_prompt = None
            print(f"user> {user_prompt}")
        else:
            try:
                user_prompt = input("user> ").strip()
            except EOFError:
                print()
                return 0

        if not user_prompt:
            if interactive:
                continue
            return 0

        if user_prompt.lower() in {"exit", "quit"}:
            return 0

        result = agent.run_sync(user_prompt, message_history=history, deps=deps)
        history = result.all_messages()
        output = result.output
        recovered_result = extract_tool_result(result)

        if isinstance(output, SimulationRunResult):
            print_result(output, args.json)
            if not interactive:
                return 0
        elif recovered_result is not None:
            print_result(recovered_result, args.json)
            if not interactive:
                return 0
        else:
            print(f"agent> {output}")
            if not interactive:
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
