"""
Simple string input example

The most basic way to use the SDK - just pass a string prompt.
"""

from dotenv import load_dotenv

from edgee import Edgee

load_dotenv()

edgee = Edgee()

response = edgee.send(
    model="devstral2",
    input="What is the capital of France?",
)

print(f"Content: {response.text}")
print(f"Usage: {response.usage}")
