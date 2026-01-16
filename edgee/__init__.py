"""Edgee Gateway SDK for Python"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ValidationError

# API Configuration
DEFAULT_BASE_URL = "https://api.edgee.ai"
API_ENDPOINT = "/v1/chat/completions"


@dataclass
class FunctionDefinition:
    name: str
    description: str | None = None
    parameters: dict | None = None


@dataclass
class OpenAITool:
    """OpenAI tool format for API requests."""

    type: str  # "function"
    function: FunctionDefinition


@dataclass
class ToolCall:
    id: str
    type: str
    function: dict  # {"name": str, "arguments": str}


class Tool:
    """Tool with schema validation and automatic execution.

    Use this class to define tools that can be automatically executed
    during the agentic loop when using simple send mode.

    Args:
        name: The name of the tool (must be unique)
        schema: A Pydantic BaseModel class defining the tool's parameters
        handler: A callable that takes the validated parameters and returns a result
        description: Optional description of what the tool does

    Example:
        ```python
        from pydantic import BaseModel
        from edgee import Tool

        class WeatherParams(BaseModel):
            location: str

        def get_weather(params: WeatherParams) -> dict:
            return {"temperature": 22, "location": params.location}

        weather_tool = Tool(
            name="get_weather",
            description="Get the current weather for a location",
            schema=WeatherParams,
            handler=get_weather,
        )
        ```
    """

    def __init__(
        self,
        name: str,
        schema: type[BaseModel],
        handler: Callable[..., Any],
        description: str | None = None,
    ):
        self.name = name
        self.description = description
        self.schema = schema
        self.handler = handler

    def to_dict(self) -> dict:
        """Convert to OpenAI tool format for API requests."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema.model_json_schema(),
            },
        }

    def execute(self, args: dict) -> Any:
        """Validate arguments with Pydantic and execute the handler.

        Args:
            args: Dictionary of arguments to validate and pass to the handler

        Returns:
            The result from the handler function

        Raises:
            ValidationError: If the arguments don't match the schema
        """
        validated = self.schema.model_validate(args)
        return self.handler(validated)


def create_tool(
    name: str,
    schema: type[BaseModel],
    handler: Callable[..., Any],
    description: str | None = None,
) -> Tool:
    """Helper function to create a Tool instance.

    This is an alternative to using the Tool class directly.

    Args:
        name: The name of the tool
        schema: A Pydantic BaseModel class defining the tool's parameters
        handler: A callable that takes the validated parameters and returns a result
        description: Optional description of what the tool does

    Returns:
        A configured Tool instance
    """
    return Tool(name=name, schema=schema, handler=handler, description=description)


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass
class InputObject:
    messages: list[dict]
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


@dataclass
class Choice:
    index: int
    message: dict
    finish_reason: str | None


@dataclass
class InputTokenDetails:
    cached_tokens: int


@dataclass
class OutputTokenDetails:
    reasoning_tokens: int


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_tokens_details: InputTokenDetails
    output_tokens_details: OutputTokenDetails


@dataclass
class SendResponse:
    choices: list[Choice]
    usage: Usage | None = None

    @property
    def text(self) -> str | None:
        """Convenience property to get text content from the first choice."""
        if self.choices and self.choices[0].message.get("content"):
            return self.choices[0].message["content"]
        return None

    @property
    def message(self) -> dict | None:
        """Convenience property to get the message from the first choice."""
        if self.choices:
            return self.choices[0].message
        return None

    @property
    def finish_reason(self) -> str | None:
        """Convenience property to get finish_reason from the first choice."""
        if self.choices and self.choices[0].finish_reason:
            return self.choices[0].finish_reason
        return None

    @property
    def tool_calls(self) -> list | None:
        """Convenience property to get tool_calls from the first choice."""
        if self.choices and self.choices[0].message.get("tool_calls"):
            return self.choices[0].message["tool_calls"]
        return None


@dataclass
class StreamToolCallDelta:
    """Partial tool call in a stream."""

    index: int
    id: str | None = None
    type: str | None = None
    function: dict | None = None  # {"name": str, "arguments": str}


@dataclass
class StreamDelta:
    role: str | None = None
    content: str | None = None
    tool_calls: list[dict] | None = None


@dataclass
class StreamChoice:
    index: int
    delta: StreamDelta
    finish_reason: str | None = None


@dataclass
class StreamChunk:
    choices: list[StreamChoice]

    @property
    def text(self) -> str | None:
        """Convenience property to get text content from the first choice."""
        if self.choices and self.choices[0].delta.content:
            return self.choices[0].delta.content
        return None

    @property
    def role(self) -> str | None:
        """Convenience property to get role from the first choice."""
        if self.choices and self.choices[0].delta.role:
            return self.choices[0].delta.role
        return None

    @property
    def finish_reason(self) -> str | None:
        """Convenience property to get finish_reason from the first choice."""
        if self.choices and self.choices[0].finish_reason:
            return self.choices[0].finish_reason
        return None

    @property
    def tool_call_deltas(self) -> list[dict] | None:
        """Get tool call deltas from the first choice."""
        if self.choices and self.choices[0].delta.tool_calls:
            return self.choices[0].delta.tool_calls
        return None


# Stream events for tool-enabled streaming
@dataclass
class ChunkEvent:
    """A chunk of streamed content."""

    type: str = "chunk"
    chunk: StreamChunk = None


@dataclass
class ToolStartEvent:
    """Tool execution is starting."""

    type: str = "tool_start"
    tool_call: dict = None


@dataclass
class ToolResultEvent:
    """Tool execution completed."""

    type: str = "tool_result"
    tool_call_id: str = None
    tool_name: str = None
    result: Any = None


@dataclass
class IterationCompleteEvent:
    """One iteration of the tool loop completed."""

    type: str = "iteration_complete"
    iteration: int = 0


StreamEvent = ChunkEvent | ToolStartEvent | ToolResultEvent | IterationCompleteEvent


@dataclass
class EdgeeConfig:
    api_key: str | None = None
    base_url: str | None = None


class Edgee:
    def __init__(
        self,
        config: str | EdgeeConfig | dict | None = None,
    ):
        if isinstance(config, str):
            # Backward compatibility: accept api_key as string
            api_key = config
            base_url = None
        elif isinstance(config, EdgeeConfig):
            api_key = config.api_key
            base_url = config.base_url
        elif isinstance(config, dict):
            api_key = config.get("api_key")
            base_url = config.get("base_url")
        else:
            api_key = None
            base_url = None

        self.api_key = api_key or os.environ.get("EDGEE_API_KEY", "")
        if not self.api_key:
            raise ValueError("EDGEE_API_KEY is not set")

        self.base_url = base_url or os.environ.get("EDGEE_BASE_URL", DEFAULT_BASE_URL)

    def send(
        self,
        model: str,
        input: str | InputObject | dict,
        tools: list[Tool] | None = None,
        max_tool_iterations: int = 10,
        stream: bool = False,
    ):
        """Send a completion request to the Edgee AI Gateway.

        Args:
            model: The model to use for completion
            input: The input (string, dict, or InputObject)
            tools: Optional list of Tool instances for automatic execution (simple mode only)
            max_tool_iterations: Maximum number of tool execution iterations (default: 10)
            stream: If True, returns a generator yielding StreamChunk objects.
                   If False, returns a SendResponse object.

        Returns:
            SendResponse if stream=False, or a generator yielding StreamChunk objects if stream=True.

        Note:
            When using string input with tools, the SDK will automatically:
            1. Convert tools to OpenAI format
            2. Execute tools when the model requests them
            3. Send tool results back to the model
            4. Loop until the model returns a final response or max iterations reached
        """
        # Simple mode: string input with optional tools for automatic execution
        if isinstance(input, str):
            if tools:
                # Agentic loop mode: auto-execute tools
                return self._send_simple(model, input, tools, max_tool_iterations)
            else:
                # Simple string without tools
                messages = [{"role": "user", "content": input}]
                return self._call_api(model, messages, stream=stream)

        # Advanced mode: full InputObject or dict (manual tool handling)
        if isinstance(input, InputObject):
            messages = input.messages
            api_tools = input.tools
            tool_choice = input.tool_choice
        else:
            messages = input.get("messages", [])
            api_tools = input.get("tools")
            tool_choice = input.get("tool_choice")

        return self._call_api(
            model, messages, api_tools=api_tools, tool_choice=tool_choice, stream=stream
        )

    def _send_simple(
        self,
        model: str,
        input: str,
        tools: list[Tool],
        max_iterations: int,
    ) -> SendResponse:
        """Handle simple mode with automatic tool execution (agentic loop).

        Args:
            model: The model to use
            input: The user's input string
            tools: List of Tool instances to use
            max_iterations: Maximum number of iterations

        Returns:
            Final SendResponse after all tool executions complete
        """
        messages: list[dict] = [{"role": "user", "content": input}]
        openai_tools = [t.to_dict() for t in tools]
        tool_map = {t.name: t for t in tools}

        total_usage: Usage | None = None

        for _ in range(max_iterations):
            response = self._call_api(model, messages, api_tools=openai_tools)

            # Accumulate usage
            if response.usage:
                if total_usage is None:
                    total_usage = Usage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                        input_tokens_details=InputTokenDetails(
                            cached_tokens=response.usage.input_tokens_details.cached_tokens
                        ),
                        output_tokens_details=OutputTokenDetails(
                            reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens
                        ),
                    )
                else:
                    total_usage.prompt_tokens += response.usage.prompt_tokens
                    total_usage.completion_tokens += response.usage.completion_tokens
                    total_usage.total_tokens += response.usage.total_tokens
                    total_usage.input_tokens_details.cached_tokens += (
                        response.usage.input_tokens_details.cached_tokens
                    )
                    total_usage.output_tokens_details.reasoning_tokens += (
                        response.usage.output_tokens_details.reasoning_tokens
                    )

            # No tool calls? We're done - return final response
            if not response.tool_calls:
                return SendResponse(choices=response.choices, usage=total_usage)

            # Add assistant's message (with tool_calls) to messages
            messages.append(response.message)

            # Execute each tool call and add results
            for tool_call in response.tool_calls:
                tool_name = tool_call["function"]["name"]
                tool = tool_map.get(tool_name)

                if tool:
                    try:
                        raw_args = json.loads(tool_call["function"]["arguments"])
                        result = tool.execute(raw_args)
                    except ValidationError as e:
                        result = {"error": f"Invalid arguments: {e}"}
                    except json.JSONDecodeError as e:
                        result = {"error": f"Failed to parse arguments: {e}"}
                    except Exception as e:
                        result = {"error": f"Tool execution failed: {e}"}
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result if isinstance(result, str) else json.dumps(result),
                    }
                )

            # Loop continues - model will process tool results

        # Max iterations reached
        raise RuntimeError(f"Max tool iterations ({max_iterations}) reached")

    def _call_api(
        self,
        model: str,
        messages: list[dict],
        api_tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stream: bool = False,
    ) -> SendResponse:
        """Make an API call to the Edgee AI Gateway.

        Args:
            model: The model to use
            messages: List of message dicts
            api_tools: Optional list of tools in OpenAI format
            tool_choice: Optional tool choice configuration
            stream: Whether to stream the response

        Returns:
            SendResponse object
        """
        body: dict = {"model": model, "messages": messages}
        if stream:
            body["stream"] = True
        if api_tools:
            body["tools"] = api_tools
        if tool_choice:
            body["tool_choice"] = tool_choice

        request = Request(
            f"{self.base_url}{API_ENDPOINT}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        if stream:
            return self._handle_streaming_response(request)
        else:
            return self._handle_non_streaming_response(request)

    def _handle_non_streaming_response(self, request: Request) -> SendResponse:
        """Handle non-streaming response."""
        try:
            with urlopen(request) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"API error {e.code}: {error_body}") from e

        choices = [
            Choice(
                index=c["index"],
                message=c["message"],
                finish_reason=c.get("finish_reason"),
            )
            for c in data["choices"]
        ]

        usage = None
        if "usage" in data:
            usage_data = data["usage"]
            usage = Usage(
                prompt_tokens=usage_data["prompt_tokens"],
                completion_tokens=usage_data["completion_tokens"],
                total_tokens=usage_data["total_tokens"],
                input_tokens_details=InputTokenDetails(
                    cached_tokens=usage_data.get("input_tokens_details", {}).get("cached_tokens", 0)
                ),
                output_tokens_details=OutputTokenDetails(
                    reasoning_tokens=usage_data.get("output_tokens_details", {}).get(
                        "reasoning_tokens", 0
                    )
                ),
            )

        return SendResponse(choices=choices, usage=usage)

    def _handle_streaming_response(self, request: Request):
        """Handle streaming response, yielding StreamChunk objects."""
        try:
            with urlopen(request) as response:
                # Read and parse SSE stream
                for line in response:
                    decoded_line = line.decode("utf-8")

                    if decoded_line.strip() == "":
                        continue

                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:].strip()

                        # Check for stream end signal
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)

                            # Parse choices
                            choices = []
                            for c in data.get("choices", []):
                                delta_data = c.get("delta", {})
                                delta = StreamDelta(
                                    role=delta_data.get("role"),
                                    content=delta_data.get("content"),
                                    tool_calls=delta_data.get("tool_calls"),
                                )
                                choice = StreamChoice(
                                    index=c["index"],
                                    delta=delta,
                                    finish_reason=c.get("finish_reason"),
                                )
                                choices.append(choice)

                            yield StreamChunk(choices=choices)
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue

        except HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"API error {e.code}: {error_body}") from e

    def stream(
        self,
        model: str,
        input: str | InputObject | dict,
        tools: list[Tool] | None = None,
        max_tool_iterations: int = 10,
    ):
        """Stream a completion request from the Edgee AI Gateway.

        Args:
            model: The model to use for completion
            input: The input (string, dict, or InputObject)
            tools: Optional list of Tool instances for automatic execution (simple mode only)
            max_tool_iterations: Maximum number of tool execution iterations (default: 10)

        Yields:
            StreamChunk objects if no tools provided.
            StreamEvent objects (ChunkEvent, ToolStartEvent, ToolResultEvent, IterationCompleteEvent)
            if tools are provided.

        Example without tools:
            ```python
            for chunk in edgee.stream("gpt-4o", "Hello!"):
                print(chunk.text, end="")
            ```

        Example with tools:
            ```python
            for event in edgee.stream("gpt-4o", "What's the weather?", tools=[weather_tool]):
                if event.type == "chunk":
                    print(event.chunk.text, end="")
                elif event.type == "tool_result":
                    print(f"Tool result: {event.result}")
            ```
        """
        # Simple mode with tools: use agentic streaming loop
        if isinstance(input, str) and tools:
            return self._stream_simple(model, input, tools, max_tool_iterations)

        # Simple mode without tools or advanced mode: regular streaming
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
            return self._call_api(model, messages, stream=True)

        # Advanced mode: full InputObject or dict
        if isinstance(input, InputObject):
            messages = input.messages
            api_tools = input.tools
            tool_choice = input.tool_choice
        else:
            messages = input.get("messages", [])
            api_tools = input.get("tools")
            tool_choice = input.get("tool_choice")

        return self._call_api(
            model, messages, api_tools=api_tools, tool_choice=tool_choice, stream=True
        )

    def _stream_simple(
        self,
        model: str,
        input: str,
        tools: list[Tool],
        max_iterations: int,
    ):
        """Handle simple mode streaming with automatic tool execution.

        Yields StreamEvent objects for chunks, tool starts, tool results, and iteration completion.
        """
        messages: list[dict] = [{"role": "user", "content": input}]
        openai_tools = [t.to_dict() for t in tools]
        tool_map = {t.name: t for t in tools}

        for iteration in range(1, max_iterations + 1):
            # Accumulate the full response from stream
            role: str | None = None
            content = ""
            tool_calls_accumulator: dict[int, dict] = {}

            # Stream the response
            for chunk in self._call_api(model, messages, api_tools=openai_tools, stream=True):
                # Yield the chunk as an event
                yield ChunkEvent(chunk=chunk)

                # Accumulate role
                if chunk.role:
                    role = chunk.role

                # Accumulate content
                if chunk.text:
                    content += chunk.text

                # Accumulate tool calls from deltas
                tool_call_deltas = chunk.tool_call_deltas
                if tool_call_deltas:
                    for delta in tool_call_deltas:
                        idx = delta.get("index", 0)
                        if idx in tool_calls_accumulator:
                            # Append to existing tool call
                            existing = tool_calls_accumulator[idx]
                            if delta.get("function", {}).get("arguments"):
                                existing["function"]["arguments"] += delta["function"]["arguments"]
                        else:
                            # Start new tool call
                            tool_calls_accumulator[idx] = {
                                "id": delta.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": delta.get("function", {}).get("name", ""),
                                    "arguments": delta.get("function", {}).get("arguments", ""),
                                },
                            }

            # Convert accumulated tool calls to list
            tool_calls = list(tool_calls_accumulator.values())

            # No tool calls? We're done
            if not tool_calls:
                return

            # Add assistant's message (with tool_calls) to messages
            assistant_message: dict = {
                "role": role or "assistant",
                "tool_calls": tool_calls,
            }
            if content:
                assistant_message["content"] = content
            messages.append(assistant_message)

            # Execute each tool call and add results
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool = tool_map.get(tool_name)

                # Yield tool_start event
                yield ToolStartEvent(tool_call=tool_call)

                if tool:
                    try:
                        raw_args = json.loads(tool_call["function"]["arguments"])
                        result = tool.execute(raw_args)
                    except ValidationError as e:
                        result = {"error": f"Invalid arguments: {e}"}
                    except json.JSONDecodeError as e:
                        result = {"error": f"Failed to parse arguments: {e}"}
                    except Exception as e:
                        result = {"error": f"Tool execution failed: {e}"}
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                # Yield tool_result event
                yield ToolResultEvent(
                    tool_call_id=tool_call["id"],
                    tool_name=tool_name,
                    result=result,
                )

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result if isinstance(result, str) else json.dumps(result),
                    }
                )

            # Yield iteration complete event
            yield IterationCompleteEvent(iteration=iteration)

            # Loop continues - model will process tool results

        # Max iterations reached
        raise RuntimeError(f"Max tool iterations ({max_iterations}) reached")
