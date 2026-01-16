# Edgee Python SDK

Lightweight, type-safe Python SDK for the [Edgee AI Gateway](https://www.edgee.cloud).

[![PyPI version](https://img.shields.io/pypi/v/edgee.svg)]( )
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Installation

```bash
pip install edgee
```

## Quick Start

```python
import os
from edgee import Edgee

edgee = Edgee("your-api-key")

# Send a simple request
response = edgee.send(
    model="gpt-4o",
    input="What is the capital of France?"
)

print(response.text)
# "The capital of France is Paris."
```

## Send Method

The `send()` method makes non-streaming chat completion requests:

```python
response = edgee.send(
    model="gpt-4o",
    input="Hello, world!"
)

# Access response
print(response.text)           # Text content
print(response.finish_reason)  # Finish reason
print(response.tool_calls)     # Tool calls (if any)
```

## Stream Method

The `stream()` method enables real-time streaming responses:

```python
for chunk in edgee.stream("gpt-4o", "Tell me a story"):
    if chunk.text:
        print(chunk.text, end="", flush=True)

    if chunk.finish_reason:
        print(f"\nFinished: {chunk.finish_reason}")
```

## Features

- ✅ **Type-safe** - Full type hints with dataclasses
- ✅ **OpenAI-compatible** - Works with any model supported by Edgee
- ✅ **Streaming** - Real-time response streaming with generators
- ✅ **Tool calling** - Full support for function calling
- ✅ **Flexible input** - Accept strings, dicts, or InputObject
- ✅ **Zero dependencies** - Uses only Python standard library

## Documentation

For complete documentation, examples, and API reference, visit:

**👉 [Official Python SDK Documentation](https://www.edgee.cloud/docs/sdk/python)**

The documentation includes:
- [Configuration guide](https://www.edgee.cloud/docs/sdk/python/configuration) - Multiple ways to configure the SDK
- [Send method](https://www.edgee.cloud/docs/sdk/python/send) - Complete guide to non-streaming requests
- [Stream method](https://www.edgee.cloud/docs/sdk/python/stream) - Streaming responses guide
- [Tools](https://www.edgee.cloud/docs/sdk/python/tools) - Function calling guide

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
