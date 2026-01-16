"""
Automatic tool execution (Simple mode)

Define tools with Pydantic schemas and handlers.
The SDK automatically executes tools and loops
until the model produces a final response.
"""

from dotenv import load_dotenv
from pydantic import BaseModel

from edgee import Edgee, Tool

load_dotenv()

edgee = Edgee()


# Define the tool schema using Pydantic
class WeatherParams(BaseModel):
    location: str


# Define the tool handler
def get_weather(params: WeatherParams) -> dict:
    """Simulated weather API call."""
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


# Create the tool
weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a location",
    schema=WeatherParams,
    handler=get_weather,
)

# Send request with automatic tool execution
response = edgee.send(
    model="devstral2",
    input="What's the weather like in Paris?",
    tools=[weather_tool],
)

print(f"Content: {response.text}")
print(f"Total usage: {response.usage}")
