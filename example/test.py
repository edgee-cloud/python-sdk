"""Example usage of Edgee Gateway SDK

Install the package from PyPI:
    pip install edgee
"""

from pydantic import BaseModel

from edgee import Edgee, EdgeeConfig, Tool, create_tool

edgee = Edgee(api_key="YOUR_API_KEY")
# Test 1: Simple string input
print("Test 1: Simple string input")
response1 = edgee.send(
    model="devstral2",
    input="What is the capital of France?",
)
print(f"Content: {response1.text}")
print(f"Usage: {response1.usage}")
print()

# Test 2: Full input object with messages
print("Test 2: Full input object with messages")
response2 = edgee.send(
    model="devstral2",
    input={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"},
        ],
    },
)
print(f"Content: {response2.text}")
print()

# Test 3: With tools
print("Test 3: With tools")
response3 = edgee.send(
    model="devstral2",
    input={
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                },
            },
        ],
        "tool_choice": "auto",
    },
)
print(f"Content: {response3.text}")
print(f"Tool calls: {response3.tool_calls}")
print()

# Test 4: Streaming
print("Test 4: Streaming")
for chunk in edgee.stream(model="devstral2", input="What is Python?"):
    if chunk.text:
        print(chunk.text, end="", flush=True)
print("\n")


# Test 5: Tool class with automatic execution (agentic loop)
print("Test 5: Tool class with automatic execution")


# Define the tool schema using Pydantic
class WeatherParams(BaseModel):
    location: str


# Define the tool handler
def get_weather(params: WeatherParams) -> dict:
    """Simulated weather API call."""
    # In a real app, this would call a weather API
    weather_data = {
        "Paris": {"temperature": 18, "condition": "partly cloudy"},
        "London": {"temperature": 12, "condition": "rainy"},
        "New York": {"temperature": 22, "condition": "sunny"},
    }
    data = weather_data.get(
        params.location, {"temperature": 20, "condition": "unknown"}
    )
    return {
        "location": params.location,
        "temperature": data["temperature"],
        "condition": data["condition"],
    }


# Create the tool
weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a location",
    schema=WeatherParams,
    handler=get_weather,
)

# Alternative: use create_tool helper
# weather_tool = create_tool(
#     name="get_weather",
#     description="Get the current weather for a location",
#     schema=WeatherParams,
#     handler=get_weather,
# )

# Send request with automatic tool execution
response5 = edgee.send(
    model="devtral2",
    input="What's the weather like in Paris?",
    tools=[weather_tool],
)
print(f"Content: {response5.text}")
print(f"Total usage: {response5.usage}")
print()


# Test 6: Multiple tools
print("Test 6: Multiple tools")


class CalculatorParams(BaseModel):
    operation: str  # "add", "subtract", "multiply", "divide"
    a: float
    b: float


def calculate(params: CalculatorParams) -> dict:
    """Simple calculator."""
    operations = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else "Error: division by zero",
    }
    op = operations.get(params.operation)
    if op:
        result = op(params.a, params.b)
        return {"operation": params.operation, "a": params.a, "b": params.b, "result": result}
    return {"error": f"Unknown operation: {params.operation}"}


calculator_tool = Tool(
    name="calculate",
    description="Perform basic arithmetic operations (add, subtract, multiply, divide)",
    schema=CalculatorParams,
    handler=calculate,
)

response6 = edgee.send(
    model="devtral2",
    input="What's 25 multiplied by 4, and then what's the weather in London?",
    tools=[weather_tool, calculator_tool],
)
print(f"Content: {response6.text}")
print(f"Total usage: {response6.usage}")
print()
