import os
import json
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import importlib
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def call(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        pass

    @staticmethod
    def _build_openai_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _tool_specs(tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        return {tool.get("name", ""): tool for tool in tools or []}

    @staticmethod
    def _safe_json_loads(arguments: Any) -> Dict[str, Any]:
        if not arguments:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _execute_tool_handler(handler_path: Optional[str], arguments: Dict[str, Any]) -> Any:
        if not handler_path:
            return "Tool handler not configured."
        module_path, function_name = handler_path.rsplit(".", 1)
        project_root = Path(__file__).parent.parent.parent
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            # Tool handlers can live in the repo-root tools/ package.
            sys.path.insert(0, project_root_str)
        module = importlib.import_module(module_path)
        handler = getattr(module, function_name)
        return handler(**arguments)

    @staticmethod
    def _anthropic_block_to_dict(block: Any) -> Dict[str, Any]:
        if isinstance(block, dict):
            return block
        block_dict = {"type": getattr(block, "type", None)}
        for key in ("text", "id", "name", "input"):
            value = getattr(block, key, None)
            if value is not None:
                block_dict[key] = value
        return block_dict

    def _run_openai_tool_loop(
        self,
        client: Any,
        model: str,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        call_kwargs: Dict[str, Any] = {"model": model}
        openai_tools = self._build_openai_tools(tools)
        if openai_tools:
            call_kwargs["tools"] = openai_tools
        call_kwargs.update(kwargs)
        messages = [{"role": "user", "content": prompt}]
        tool_specs = self._tool_specs(tools)

        for _ in range(3):
            response = client.chat.completions.create(messages=messages, **call_kwargs)
            message = response.choices[0].message
            content = getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", None) or []

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    tool_spec = tool_specs.get(tc.function.name, {})
                    handler_path = tool_spec.get("implementation", {}).get("handler")
                    arguments = self._safe_json_loads(tc.function.arguments)
                    tool_result = self._execute_tool_handler(handler_path, arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(tool_result),
                        }
                    )
                continue

            return content or ""

        return ""

    def _run_anthropic_tool_loop(
        self,
        client: Any,
        model: str,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", 4096),
        }
        if tools:
            call_kwargs["tools"] = [
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("input_schema", {}),
                }
                for tool in tools
            ]
        call_kwargs.update(kwargs)
        messages = [{"role": "user", "content": prompt}]
        tool_specs = self._tool_specs(tools)

        for _ in range(3):
            response = client.messages.create(messages=messages, **call_kwargs)
            content_blocks = list(getattr(response, "content", []) or [])
            tool_uses = [b for b in content_blocks if getattr(b, "type", None) == "tool_use"]
            text_parts = [getattr(b, "text", "") for b in content_blocks if getattr(b, "type", None) == "text"]

            if tool_uses:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [self._anthropic_block_to_dict(b) for b in content_blocks],
                    }
                )
                for tool_use in tool_uses:
                    tool_name = getattr(tool_use, "name", "")
                    tool_spec = tool_specs.get(tool_name, {})
                    handler_path = tool_spec.get("implementation", {}).get("handler")
                    arguments = self._safe_json_loads(getattr(tool_use, "input", {}))
                    tool_result = self._execute_tool_handler(handler_path, arguments)
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": getattr(tool_use, "id", ""),
                                    "content": str(tool_result),
                                }
                            ],
                        }
                    )
                continue

            return "".join(text_parts).strip()

        return ""

    def _run_gemini_tool_loop(
        self,
        client: Any,
        model: str,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        types = importlib.import_module("google.genai.types")

        # Build function declarations for the new SDK
        gemini_tools = None
        if tools:
            declarations = []
            for tool in tools:
                schema = tool.get("input_schema", {})
                properties = {}
                for prop_name, prop_schema in schema.get("properties", {}).items():
                    properties[prop_name] = types.Schema(
                        type=self._python_type_to_gemini_type(prop_schema.get("type", "string")),
                        description=prop_schema.get("description", ""),
                    )
                declarations.append(
                    types.FunctionDeclaration(
                        name=tool.get("name", ""),
                        description=tool.get("description", ""),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties=properties,
                            required=schema.get("required", []),
                        ),
                    )
                )
            gemini_tools = [types.Tool(function_declarations=declarations)]

        tool_specs = self._tool_specs(tools)

        # Build initial contents
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

        for _ in range(3):
            config = types.GenerateContentConfig(tools=gemini_tools) if gemini_tools else None
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                return ""

            parts = list(candidate.content.parts or [])
            function_calls = [p for p in parts if p.function_call is not None]
            text_parts = [p.text for p in parts if p.text]

            if function_calls:
                # Append the model's response to contents
                contents.append(types.Content(role="model", parts=parts))

                # Execute tools and collect results
                result_parts = []
                for part in function_calls:
                    fc = part.function_call
                    tool_name = fc.name
                    tool_spec = tool_specs.get(tool_name, {})
                    handler_path = tool_spec.get("implementation", {}).get("handler")
                    arguments = dict(fc.args) if fc.args else {}
                    tool_result = self._execute_tool_handler(handler_path, arguments)
                    result_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response={"result": str(tool_result)},
                            )
                        )
                    )

                contents.append(types.Content(role="user", parts=result_parts))
                continue

            return "".join(text_parts).strip()

        return ""

    def _python_type_to_gemini_type(self, python_type: str) -> Any:
        try:
            types = importlib.import_module("google.genai.types")
            return {
                "string": types.Type.STRING,
                "number": types.Type.NUMBER,
                "integer": types.Type.INTEGER,
                "boolean": types.Type.BOOLEAN,
                "array": types.Type.ARRAY,
                "object": types.Type.OBJECT,
            }.get(python_type, types.Type.STRING)
        except Exception:
            return None


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        try:
            openai = importlib.import_module("openai")
            client_kwargs = {"api_key": self.api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def call(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        try:
            return self._run_openai_tool_loop(self.client, self.model, prompt, tools, **kwargs)
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")


class LMStudioProvider(LLMProvider):
    """LM Studio OpenAI-compatible local LLM provider."""

    def __init__(self, model: str = "local-model", base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.base_url = base_url or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        self.api_key = api_key or os.getenv("LMSTUDIO_API_KEY", "lm-studio")
        try:
            openai = importlib.import_module("openai")
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def call(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        try:
            return self._run_openai_tool_loop(self.client, self.model, prompt, tools, **kwargs)
        except Exception as e:
            raise RuntimeError(f"LM Studio API call failed: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-5"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        try:
            anthropic = importlib.import_module("anthropic")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def call(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        try:
            return self._run_anthropic_tool_loop(self.client, self.model, prompt, tools, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}")


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        try:
            ollama = importlib.import_module("ollama")
            self.client = ollama
        except ImportError:
            raise ImportError("ollama package not installed. Run: pip install ollama")

    def call(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        try:
            full_prompt = prompt
            if tools:
                tools_description = "\n\nAvailable tools:\n"
                for tool in tools:
                    tools_description += f"- {tool.get('name')}: {tool.get('description')}\n"
                full_prompt += tools_description
            response = self.client.generate(model=self.model, prompt=full_prompt, stream=False, **kwargs)
            return response.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}")


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider using the new google-genai SDK."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        try:
            genai = importlib.import_module("google.genai")
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")

    def call(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        try:
            return self._run_gemini_tool_loop(self.client, self.model, prompt, tools, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")


def get_llm_provider(provider_name: Optional[str] = None, **config) -> LLMProvider:
    provider_name = provider_name or os.getenv("LLM_PROVIDER", "openai")
    provider_name = provider_name.lower()
    providers = {
        "openai": OpenAIProvider,
        "lmstudio": LMStudioProvider,
        "lm-studio": LMStudioProvider,
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,
        "ollama": OllamaProvider,
        "gemini": GeminiProvider,
        "google": GeminiProvider,
    }
    if provider_name not in providers:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Supported providers: {', '.join(providers.keys())}"
        )
    return providers[provider_name](**config)


def call_llm(
    prompt: str,
    provider_name: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> str:
    provider = get_llm_provider(provider_name, **kwargs)
    return provider.call(prompt, tools=tools)
