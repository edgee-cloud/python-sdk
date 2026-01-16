"""
Simple streaming without tools

Stream responses token by token for real-time output.
"""

from dotenv import load_dotenv

from edgee import Edgee

load_dotenv()

edgee = Edgee()

print("Response: ", end="", flush=True)

for chunk in edgee.stream(model="devstral2", input="Say hello in 10 words!"):
    if chunk.text:
        print(chunk.text, end="", flush=True)

print()
