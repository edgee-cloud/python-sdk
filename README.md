# Edgee Gateway SDK

Lightweight Python SDK for Edgee AI Gateway.

## Installation

```bash
pip install edgee
```

## Usage

```python
from edgee import Edgee

edgee = Edgee(os.environ.get("EDGEE_API_KEY"))
```

### Simple Input

```python
response = edgee.send(
    model="gpt-4o",
    input="What is the capital of France?",
)

print(response.text)
```

### Full Input with Messages

```python
response = edgee.send(
    model="gpt-4o",
    input={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    },
)
```

### With Tools

```python
response = edgee.send(
    model="gpt-4o",
    input={
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                },
            },
        ],
        "tool_choice": "auto",
    },
)

if response.tool_calls:
    print(response.tool_calls)
```

### Streaming

#### Simple Text Streaming

The simplest way to stream text responses:

```python
for text in edgee.stream_text(model="gpt-4o", input="Tell me a story"):
    print(text, end="", flush=True)
```

#### Streaming with More Control

Access chunk properties when you need more control:

```python
for chunk in edgee.stream(model="gpt-4o", input="Tell me a story"):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

#### Alternative: Using send(stream=True)

```python
for chunk in edgee.send(model="gpt-4o", input="Tell me a story", stream=True):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

#### Accessing Full Chunk Data

When you need complete access to the streaming response:

```python
for chunk in edgee.stream(model="gpt-4o", input="Hello"):
    if chunk.role:
        print(f"Role: {chunk.role}")
    if chunk.text:
        print(chunk.text, end="", flush=True)
    if chunk.finish_reason:
        print(f"\nFinish: {chunk.finish_reason}")
```

## Response

```python
@dataclass
class SendResponse:
    choices: list[Choice]
    usage: Optional[Usage]

    # Convenience properties for easy access
    text: str | None  # Shortcut for choices[0].message["content"]
    message: dict | None  # Shortcut for choices[0].message
    finish_reason: str | None  # Shortcut for choices[0].finish_reason
    tool_calls: list | None  # Shortcut for choices[0].message["tool_calls"]

@dataclass
class Choice:
    index: int
    message: dict  # {"role": str, "content": str | None, "tool_calls": list | None}
    finish_reason: str | None

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

### Streaming Response

```python
@dataclass
class StreamChunk:
    choices: list[StreamChoice]

    # Convenience properties for easy access
    text: str | None  # Shortcut for choices[0].delta.content
    role: str | None  # Shortcut for choices[0].delta.role
    finish_reason: str | None  # Shortcut for choices[0].finish_reason

@dataclass
class StreamChoice:
    index: int
    delta: StreamDelta
    finish_reason: str | None

@dataclass
class StreamDelta:
    role: str | None  # Only present in first chunk
    content: str | None
    tool_calls: list[dict] | None
```

To learn more about this SDK, please refer to the [dedicated documentation](https://www.edgee.cloud/docs/sdk/python).