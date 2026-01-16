"""
Multiple tools with automatic execution

Pass multiple tools and the model will decide
which ones to call based on the user's request.
"""

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
    operation: str  # "add", "subtract", "multiply", "divide"
    a: float
    b: float


def calculate(params: CalculatorParams) -> dict:
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

# Use both tools
response = edgee.send(
    model="devstral2",
    input="What's 25 multiplied by 4, and then what's the weather in London?",
    tools=[weather_tool, calculator_tool],
)

print(f"Content: {response.text}")
print(f"Total usage: {response.usage}")
