"""
Full input object with messages

Use the messages array for multi-turn conversations
and system prompts.
"""

from dotenv import load_dotenv

from edgee import Edgee

load_dotenv()

edgee = Edgee()

response = edgee.send(
    model="devstral2",
    input={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"},
        ],
    },
)

print(f"Content: {response.text}")
