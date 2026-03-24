from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from deeplense_agent.agent import build_deeplense_agent, AgentDependencies
from deeplense_agent.simulator import DeepLenseSimulationService

service = DeepLenseSimulationService(output_root="artifacts/trace_test")
agent = build_deeplense_agent(
    simulation_service=service,
    provider="ollama",
    ollama_model="qwen3:8b",
)
deps = AgentDependencies(simulation_service=service)

result1 = agent.run_sync(
    "Generate 2 Model III no_sub images at 64 pixels with seed 123 and label the run hst-baseline.",
    deps=deps,
)

result2 = agent.run_sync(
    "Yes, go ahead and run it.",
    message_history=result1.all_messages(),
    deps=deps,
)

print("\nRAW TOOL CALLS")
for msg in result2.all_messages():
    if isinstance(msg, ModelResponse):
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                print(part.tool_name, part.args_as_dict())

print("\nTOOL RETURNS")
for msg in result2.all_messages():
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                print("tool:", part.tool_name)
                print(part.content)

print("\nFINAL RESOLVED REQUEST")
for msg in result2.all_messages():
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == "run_deeplense_simulation":
                print(part.content.plan.resolved_request.model_dump(by_alias=True))
