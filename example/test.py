"""Example usage of Edgee Gateway SDK"""

import os
import sys

# Add parent directory to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgee import Edgee

edgee = Edgee(os.environ.get("EDGEE_API_KEY", "test-key"))

# Test 1: Simple string input
print("Test 1: Simple string input")
response1 = edgee.send(
    model="mistral/mistral-small-latest",
    input="What is the capital of France?",
)
print(f"Content: {response1.choices[0].message['content']}")
print(f"Usage: {response1.usage}")
print()

# Test 2: Full input object with messages
print("Test 2: Full input object with messages")
response2 = edgee.send(
    model="mistral/mistral-small-latest",
    input={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"},
        ],
    },
)
print(f"Content: {response2.choices[0].message['content']}")
print()

# Test 3: With tools
#print("Test 3: With tools")
#response3 = edgee.send(
#    model="gpt-4o",
#    input={
#        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
#        "tools": [
#            {
#                "type": "function",
#                "function": {
#                    "name": "get_weather",
#                    "description": "Get the current weather for a location",
#                    "parameters": {
#                        "type": "object",
#                        "properties": {
#                            "location": {"type": "string", "description": "City name"},
#                        },
#                        "required": ["location"],
#                    },
#                },
#            },
#        ],
#        "tool_choice": "auto",
#    },
#)
#print(f"Content: {response3.choices[0].message.get('content')}")
#print(f"Tool calls: {response3.choices[0].message.get('tool_calls')}")
#print()

# Test 4: Streaming
print("Test 4: Streaming")
for chunk in edgee.stream(
    model="mistral/mistral-small-latest",
    input="Tell me a short story about a robot",
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

# Test 5: Streaming with messages
print("Test 5: Streaming with system message")
for chunk in edgee.stream(
    model="mistral/mistral-small-latest",
    input={
        "messages": [
            {"role": "system", "content": "You are a poetic assistant. Respond in rhyme."},
            {"role": "user", "content": "What is Python?"},
        ],
    },
):
    delta = chunk.choices[0].delta
    if delta.role:
        print(f"[Role: {delta.role}]")
    if delta.content:
        print(delta.content, end="", flush=True)
    if chunk.choices[0].finish_reason:
        print(f"\n[Finished: {chunk.choices[0].finish_reason}]")
print()
