"""Edgee Gateway SDK for Python"""

import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# API Configuration
DEFAULT_BASE_URL = "https://api.edgee.ai"
API_ENDPOINT = "/v1/chat/completions"


@dataclass
class FunctionDefinition:
    name: str
    description: str | None = None
    parameters: dict | None = None


@dataclass
class Tool:
    type: str  # "function"
    function: FunctionDefinition


@dataclass
class ToolCall:
    id: str
    type: str
    function: dict  # {"name": str, "arguments": str}


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
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class SendResponse:
    choices: list[Choice]
    usage: Usage | None = None


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
    ) -> SendResponse:
        """Send a completion request to the Edgee AI Gateway."""

        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
            tools = None
            tool_choice = None
        elif isinstance(input, InputObject):
            messages = input.messages
            tools = input.tools
            tool_choice = input.tool_choice
        else:
            messages = input.get("messages", [])
            tools = input.get("tools")
            tool_choice = input.get("tool_choice")

        body: dict = {"model": model, "messages": messages}
        if tools:
            body["tools"] = tools
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
            usage = Usage(
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
                total_tokens=data["usage"]["total_tokens"],
            )

        return SendResponse(choices=choices, usage=usage)

    def stream(
        self,
        model: str,
        input: str | InputObject | dict,
    ):
        """Stream a completion request from the Edgee AI Gateway.

        Yields StreamChunk objects as they arrive from the API.
        """

        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
            tools = None
            tool_choice = None
        elif isinstance(input, InputObject):
            messages = input.messages
            tools = input.tools
            tool_choice = input.tool_choice
        else:
            messages = input.get("messages", [])
            tools = input.get("tools")
            tool_choice = input.get("tool_choice")

        body: dict = {"model": model, "messages": messages, "stream": True}
        if tools:
            body["tools"] = tools
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

        try:
            with urlopen(request) as response:
                # Read and parse SSE stream
                buffer = ""
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
