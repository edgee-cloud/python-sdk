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

print(response.choices[0].message["content"])
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

if response.choices[0].message.get("tool_calls"):
    print(response.choices[0].message["tool_calls"])
```

## Response

```python
@dataclass
class SendResponse:
    choices: list[Choice]
    usage: Optional[Usage]

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

To learn more about this SDK, please refer to the [dedicated documentation](https://www.edgee.cloud/docs/sdk/python).