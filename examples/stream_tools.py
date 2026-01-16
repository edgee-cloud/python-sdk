"""
Streaming with automatic tool execution

Combine streaming with auto tool execution.
You receive events for chunks, tool starts, and tool results.
"""

import json

from dotenv import load_dotenv
from pydantic import BaseModel

from edgee import Edgee, Tool

load_dotenv()

edgee = Edgee()


# Weather tool
class WeatherParams(BaseModel):
    location: str


def get_weather(params: WeatherParams) -> dict:
    weather_data = {
        "Paris": {"temperature": 18, "condition": "partly cloudy"},
        "London": {"temperature": 12, "condition": "rainy"},
        "New York": {"temperature": 22, "condition": "sunny"},
    }
    data = weather_data.get(params.location, {"temperature": 20, "condition": "unknown"})
    return {
        "location": params.location,
        "temperature": data["temperature"],
        "condition": data["condition"],
    }


weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a location",
    schema=WeatherParams,
    handler=get_weather,
)


# Calculator tool
class CalculatorParams(BaseModel):
    operation: str
    a: float
    b: float


def calculate(params: CalculatorParams) -> dict:
    ops = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else "Error: division by zero",
    }
    op = ops.get(params.operation)
    if op:
        return {
            "operation": params.operation,
            "a": params.a,
            "b": params.b,
            "result": op(params.a, params.b),
        }
    return {"error": f"Unknown operation: {params.operation}"}


calculator_tool = Tool(
    name="calculate",
    description="Perform basic arithmetic operations",
    schema=CalculatorParams,
    handler=calculate,
)

print("Streaming with tools...\n")
print("Response: ", end="", flush=True)

for event in edgee.stream(
    model="devstral2",
    input="What's 15 multiplied by 7, and what's the weather in Paris?",
    tools=[weather_tool, calculator_tool],
):
    if event.type == "chunk":
        # Stream content as it arrives
        if event.chunk.text:
            print(event.chunk.text, end="", flush=True)

    elif event.type == "tool_start":
        # Tool is about to be executed
        print(f"\n  [Tool starting: {event.tool_call['function']['name']}]")

    elif event.type == "tool_result":
        # Tool finished executing
        print(f"  [Tool result: {event.tool_name} -> {json.dumps(event.result)}]")
        print("Response: ", end="", flush=True)

    elif event.type == "iteration_complete":
        # One iteration of the tool loop completed
        print(f"  [Iteration {event.iteration} complete, continuing...]")

print()
