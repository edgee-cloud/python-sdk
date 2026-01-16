"""
Manual tool handling (Advanced mode)

Define tools using raw OpenAI format and handle
tool calls manually in your code.
"""

from dotenv import load_dotenv

from edgee import Edgee

load_dotenv()

edgee = Edgee()

response = edgee.send(
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

print(f"Content: {response.text}")
print(f"Tool calls: {response.tool_calls}")

# In manual mode, you would:
# 1. Check if response.tool_calls exists
# 2. Execute the tool yourself
# 3. Send another request with the tool result
