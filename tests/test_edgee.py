"""Tests for Edgee SDK"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from edgee import Edgee, EdgeeConfig, Tool, create_tool


class TestEdgeeConstructor:
    """Test Edgee constructor"""

    def setup_method(self):
        # Clear environment variables before each test
        os.environ.pop("EDGEE_API_KEY", None)
        os.environ.pop("EDGEE_BASE_URL", None)

    def test_with_string_api_key(self):
        """Should use provided API key (backward compatibility)"""
        client = Edgee("test-api-key")
        assert isinstance(client, Edgee)

    def test_with_empty_string_raises_error(self):
        """Should throw error when empty string is provided as API key"""
        with pytest.raises(ValueError, match="EDGEE_API_KEY is not set"):
            Edgee("")

    def test_with_config_dict(self):
        """Should use provided API key and base_url from dict"""
        client = Edgee({"api_key": "test-key", "base_url": "https://custom.example.com"})
        assert isinstance(client, Edgee)

    def test_with_config_object(self):
        """Should use provided API key and base_url from EdgeeConfig"""
        config = EdgeeConfig(api_key="test-key", base_url="https://custom.example.com")
        client = Edgee(config)
        assert isinstance(client, Edgee)

    def test_with_env_api_key(self):
        """Should use EDGEE_API_KEY environment variable"""
        os.environ["EDGEE_API_KEY"] = "env-api-key"
        client = Edgee()
        assert isinstance(client, Edgee)

    def test_with_env_base_url(self):
        """Should use EDGEE_BASE_URL environment variable"""
        os.environ["EDGEE_API_KEY"] = "env-api-key"
        os.environ["EDGEE_BASE_URL"] = "https://env-base-url.example.com"
        client = Edgee()
        assert isinstance(client, Edgee)

    def test_no_api_key_raises_error(self):
        """Should throw error when no API key provided"""
        with pytest.raises(ValueError, match="EDGEE_API_KEY is not set"):
            Edgee()

    def test_empty_config_with_env(self):
        """Should use environment variables when config is empty dict"""
        os.environ["EDGEE_API_KEY"] = "env-api-key"
        client = Edgee({})
        assert isinstance(client, Edgee)


class TestEdgeeSend:
    """Test Edgee.send method"""

    def setup_method(self):
        os.environ.pop("EDGEE_API_KEY", None)
        os.environ.pop("EDGEE_BASE_URL", None)

    def _mock_response(self, data: dict):
        """Create a mock response"""
        mock = MagicMock()
        mock.read.return_value = json.dumps(data).encode("utf-8")
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    @patch("edgee.urlopen")
    def test_send_with_string_input(self, mock_urlopen):
        """Should send request with string input"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello, world!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        client = Edgee("test-api-key")
        result = client.send(model="gpt-4", input="Hello")

        assert len(result.choices) == 1
        assert result.choices[0].message["content"] == "Hello, world!"
        assert result.usage.total_tokens == 15

        # Verify the request
        call_args = mock_urlopen.call_args[0][0]
        assert call_args.full_url == "https://api.edgee.ai/v1/chat/completions"
        body = json.loads(call_args.data.decode("utf-8"))
        assert body["model"] == "gpt-4"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

    @patch("edgee.urlopen")
    def test_send_with_input_object(self, mock_urlopen):
        """Should send request with InputObject (dict)"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        client = Edgee("test-api-key")
        client.send(
            model="gpt-4",
            input={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )

        call_args = mock_urlopen.call_args[0][0]
        body = json.loads(call_args.data.decode("utf-8"))
        assert body["messages"] == [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]

    @patch("edgee.urlopen")
    def test_send_with_tools(self, mock_urlopen):
        """Should include tools when provided"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        client = Edgee("test-api-key")
        result = client.send(
            model="gpt-4",
            input={
                "messages": [{"role": "user", "content": "What is the weather?"}],
                "tools": tools,
                "tool_choice": "auto",
            },
        )

        call_args = mock_urlopen.call_args[0][0]
        body = json.loads(call_args.data.decode("utf-8"))
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"
        assert result.choices[0].message.get("tool_calls") is not None

    @patch("edgee.urlopen")
    def test_send_without_usage(self, mock_urlopen):
        """Should handle response without usage field"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        client = Edgee("test-api-key")
        result = client.send(model="gpt-4", input="Test")

        assert result.usage is None
        assert len(result.choices) == 1

    @patch("edgee.urlopen")
    def test_send_with_multiple_choices(self, mock_urlopen):
        """Should handle multiple choices in response"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "First response"},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Second response"},
                    "finish_reason": "stop",
                },
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        client = Edgee("test-api-key")
        result = client.send(model="gpt-4", input="Test")

        assert len(result.choices) == 2
        assert result.choices[0].message["content"] == "First response"
        assert result.choices[1].message["content"] == "Second response"

    @patch("edgee.urlopen")
    def test_send_with_custom_base_url(self, mock_urlopen):
        """Should use custom base_url when provided"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        custom_base_url = "https://custom-api.example.com"
        client = Edgee({"api_key": "test-key", "base_url": custom_base_url})
        client.send(model="gpt-4", input="Test")

        call_args = mock_urlopen.call_args[0][0]
        assert call_args.full_url == f"{custom_base_url}/v1/chat/completions"

    @patch("edgee.urlopen")
    def test_send_with_env_base_url(self, mock_urlopen):
        """Should use EDGEE_BASE_URL environment variable"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        env_base_url = "https://env-base-url.example.com"
        os.environ["EDGEE_BASE_URL"] = env_base_url
        client = Edgee("test-key")
        client.send(model="gpt-4", input="Test")

        call_args = mock_urlopen.call_args[0][0]
        assert call_args.full_url == f"{env_base_url}/v1/chat/completions"

    @patch("edgee.urlopen")
    def test_config_base_url_overrides_env(self, mock_urlopen):
        """Should prioritize config base_url over environment variable"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        config_base_url = "https://config-base-url.example.com"
        os.environ["EDGEE_BASE_URL"] = "https://env-base-url.example.com"
        client = Edgee({"api_key": "test-key", "base_url": config_base_url})
        client.send(model="gpt-4", input="Test")

        call_args = mock_urlopen.call_args[0][0]
        assert call_args.full_url == f"{config_base_url}/v1/chat/completions"


class TestToolClass:
    """Test Tool class"""

    def test_tool_creation(self):
        """Should create a Tool with all properties"""

        class TestParams(BaseModel):
            name: str
            count: int = 1

        def handler(params: TestParams) -> dict:
            return {"name": params.name, "count": params.count}

        tool = Tool(
            name="test_tool",
            description="A test tool",
            schema=TestParams,
            handler=handler,
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.schema == TestParams
        assert tool.handler == handler

    def test_tool_to_dict(self):
        """Should convert Tool to OpenAI format"""

        class TestParams(BaseModel):
            location: str

        tool = Tool(
            name="get_weather",
            description="Get weather for a location",
            schema=TestParams,
            handler=lambda p: {"temp": 20},
        )

        result = tool.to_dict()

        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get weather for a location"
        assert "properties" in result["function"]["parameters"]
        assert "location" in result["function"]["parameters"]["properties"]

    def test_tool_execute(self):
        """Should validate args and execute handler"""

        class CalcParams(BaseModel):
            a: int
            b: int

        def add(params: CalcParams) -> int:
            return params.a + params.b

        tool = Tool(name="add", schema=CalcParams, handler=add)
        result = tool.execute({"a": 5, "b": 3})

        assert result == 8

    def test_tool_execute_validation_error(self):
        """Should raise ValidationError for invalid args"""

        class StrictParams(BaseModel):
            value: int

        tool = Tool(
            name="test",
            schema=StrictParams,
            handler=lambda p: p.value,
        )

        with pytest.raises(ValidationError):
            tool.execute({"value": "not_an_int"})

    def test_create_tool_helper(self):
        """Should create Tool using create_tool helper"""

        class Params(BaseModel):
            x: int

        tool = create_tool(
            name="helper_test",
            description="Test helper",
            schema=Params,
            handler=lambda p: p.x * 2,
        )

        assert isinstance(tool, Tool)
        assert tool.name == "helper_test"
        assert tool.execute({"x": 5}) == 10


class TestAgenticLoop:
    """Test automatic tool execution (agentic loop)"""

    def setup_method(self):
        os.environ.pop("EDGEE_API_KEY", None)
        os.environ.pop("EDGEE_BASE_URL", None)

    def _mock_response(self, data: dict):
        """Create a mock response"""
        mock = MagicMock()
        mock.read.return_value = json.dumps(data).encode("utf-8")
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    @patch("edgee.urlopen")
    def test_simple_send_with_tool_execution(self, mock_urlopen):
        """Should execute tools automatically and return final response"""

        # First call: model requests tool call
        tool_call_response = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Second call: model returns final response
        final_response = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The weather in Paris is sunny with 22°C.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

        mock_urlopen.side_effect = [
            self._mock_response(tool_call_response),
            self._mock_response(final_response),
        ]

        class WeatherParams(BaseModel):
            location: str

        def get_weather(params: WeatherParams) -> dict:
            return {"temperature": 22, "condition": "sunny", "location": params.location}

        weather_tool = Tool(
            name="get_weather",
            description="Get weather",
            schema=WeatherParams,
            handler=get_weather,
        )

        client = Edgee("test-api-key")
        result = client.send(
            model="gpt-4",
            input="What's the weather in Paris?",
            tools=[weather_tool],
        )

        # Should return final response
        assert result.text == "The weather in Paris is sunny with 22°C."

        # Should accumulate usage
        assert result.usage.total_tokens == 45  # 15 + 30

        # Should have made 2 API calls
        assert mock_urlopen.call_count == 2

    @patch("edgee.urlopen")
    def test_simple_send_no_tool_calls(self, mock_urlopen):
        """Should return immediately if model doesn't request tools"""
        response = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "I don't need any tools."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        mock_urlopen.return_value = self._mock_response(response)

        class Params(BaseModel):
            x: int

        tool = Tool(name="unused", schema=Params, handler=lambda p: p.x)

        client = Edgee("test-api-key")
        result = client.send(model="gpt-4", input="Hello", tools=[tool])

        assert result.text == "I don't need any tools."
        assert mock_urlopen.call_count == 1

    @patch("edgee.urlopen")
    def test_simple_send_max_iterations(self, mock_urlopen):
        """Should raise error when max iterations reached"""
        # Always return tool calls
        tool_call_response = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "loop_tool", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(tool_call_response)

        class EmptyParams(BaseModel):
            pass

        tool = Tool(name="loop_tool", schema=EmptyParams, handler=lambda p: "result")

        client = Edgee("test-api-key")

        with pytest.raises(RuntimeError, match="Max tool iterations"):
            client.send(
                model="gpt-4",
                input="Loop forever",
                tools=[tool],
                max_tool_iterations=3,
            )

        # Should have made exactly 3 calls
        assert mock_urlopen.call_count == 3

    @patch("edgee.urlopen")
    def test_simple_send_unknown_tool(self, mock_urlopen):
        """Should handle unknown tool gracefully"""
        # First call: model requests unknown tool
        tool_call_response = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "unknown_tool",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        # Second call: model handles error
        final_response = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Tool not found."},
                    "finish_reason": "stop",
                }
            ],
        }

        mock_urlopen.side_effect = [
            self._mock_response(tool_call_response),
            self._mock_response(final_response),
        ]

        class Params(BaseModel):
            x: int

        tool = Tool(name="known_tool", schema=Params, handler=lambda p: p.x)

        client = Edgee("test-api-key")
        result = client.send(model="gpt-4", input="Test", tools=[tool])

        # Should send error message for unknown tool and continue
        assert result.text == "Tool not found."

        # Verify the tool result message was sent with error
        second_call = mock_urlopen.call_args_list[1][0][0]
        body = json.loads(second_call.data.decode("utf-8"))
        tool_message = next(m for m in body["messages"] if m["role"] == "tool")
        assert "Unknown tool" in tool_message["content"]
