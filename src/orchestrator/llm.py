import os
import json
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import importlib


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def call(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            tools: Optional list of tool definitions in OpenAI format
            **kwargs: Additional arguments for the specific provider
        
        Returns:
            The LLM's response as a string
        """
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
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        openai_tools = self._build_openai_tools(tools)
        if openai_tools:
            call_kwargs["tools"] = openai_tools

        call_kwargs.update(kwargs)
        messages = list(call_kwargs.pop("messages"))
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
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in tool_calls
                        ],
                    }
                )

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_spec = tool_specs.get(tool_name, {})
                    handler_path = tool_spec.get("implementation", {}).get("handler")
                    arguments = self._safe_json_loads(tool_call.function.arguments)
                    tool_result = self._execute_tool_handler(handler_path, arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
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
            "messages": [{"role": "user", "content": prompt}],
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
        messages = list(call_kwargs.pop("messages"))
        tool_specs = self._tool_specs(tools)

        for _ in range(3):
            response = client.messages.create(messages=messages, **call_kwargs)
            content_blocks = list(getattr(response, "content", []) or [])

            tool_uses = [block for block in content_blocks if getattr(block, "type", None) == "tool_use"]
            text_parts = [getattr(block, "text", "") for block in content_blocks if getattr(block, "type", None) == "text"]

            if tool_uses:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [self._anthropic_block_to_dict(block) for block in content_blocks],
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
        model: Any,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        generation_config = kwargs.pop("generation_config", None)
        gemini_tools = None
        if tools:
            gemini_tools = []
            for tool in tools:
                gemini_tools.append(
                    self.client.protos.Tool(
                        function_declarations=[
                            self.client.protos.FunctionDeclaration(
                                name=tool.get("name", ""),
                                description=tool.get("description", ""),
                                parameters=self._convert_schema_to_parameters(tool.get("input_schema", {})),
                            )
                        ]
                    )
                )

        response = model.generate_content(
            prompt,
            tools=gemini_tools if gemini_tools else None,
            generation_config=generation_config,
            **kwargs,
        )

        parts = []
        try:
            parts = list(getattr(response, "candidates", [])[0].content.parts)
        except Exception:
            parts = []

        text_parts = []
        function_calls = []
        for part in parts:
            part_type = getattr(part, "type", None)
            if part_type == "text":
                text_parts.append(getattr(part, "text", ""))
            elif part_type == "function_call":
                function_calls.append(part)

        if function_calls:
            tool_specs = self._tool_specs(tools)
            tool_results = []
            for function_call in function_calls:
                tool_name = getattr(function_call, "name", "")
                tool_spec = tool_specs.get(tool_name, {})
                handler_path = tool_spec.get("implementation", {}).get("handler")
                arguments = self._safe_json_loads(getattr(function_call, "args", {}))
                tool_results.append(f"{tool_name}: {self._execute_tool_handler(handler_path, arguments)}")

            follow_up_prompt = (
                f"{prompt}\n\nTool results:\n" + "\n".join(tool_results)
            )
            follow_up_response = model.generate_content(
                follow_up_prompt,
                generation_config=generation_config,
                **kwargs,
            )
            return getattr(follow_up_response, "text", "") or ""

        return getattr(response, "text", "") or "" or "".join(text_parts).strip()


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", base_url: Optional[str] = None):
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
        """Call OpenAI's API."""
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
        """Call LM Studio's OpenAI-compatible API."""
        try:
            return self._run_openai_tool_loop(self.client, self.model, prompt, tools, **kwargs)
        except Exception as e:
            raise RuntimeError(f"LM Studio API call failed: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
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
        """Call Anthropic's Claude API."""
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
        """Call Ollama's local API."""
        try:
            # Ollama doesn't natively support tools, so we'll append tool info to the prompt
            full_prompt = prompt
            if tools:
                tools_description = "\n\nAvailable tools:\n"
                for tool in tools:
                    tools_description += f"- {tool.get('name')}: {tool.get('description')}\n"
                full_prompt += tools_description
            
            response = self.client.generate(
                model=self.model,
                prompt=full_prompt,
                stream=False,
                **kwargs
            )
            return response.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}")


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        try:
            genai = importlib.import_module("google.generativeai")
            genai.configure(api_key=self.api_key)
            self.client = genai
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    def call(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        """Call Google Gemini API."""
        try:
            model = self.client.GenerativeModel(self.model)
            return self._run_gemini_tool_loop(model, prompt, tools, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")
    
    def _convert_schema_to_parameters(self, schema: Dict[str, Any]) -> Optional[Any]:
        """Convert JSON schema to Gemini parameters format."""
        try:
            genai = importlib.import_module("google.generativeai")
            
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # Create a simplified parameters schema for Gemini
            schema_properties = {}
            for prop_name, prop_schema in properties.items():
                schema_properties[prop_name] = {
                    "type": prop_schema.get("type", "string"),
                    "description": prop_schema.get("description", ""),
                }
            
            return genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    k: genai.protos.Schema(type=self._python_type_to_gemini_type(v.get("type", "string")))
                    for k, v in schema_properties.items()
                },
                required=required,
            )
        except Exception as e:
            print(f"Warning: Could not convert schema to Gemini parameters: {e}")
            return None
    
    def _python_type_to_gemini_type(self, python_type: str) -> Any:
        """Convert Python type string to Gemini type."""
        try:
            genai = importlib.import_module("google.generativeai")
            type_mapping = {
                "string": genai.protos.Type.STRING,
                "number": genai.protos.Type.NUMBER,
                "integer": genai.protos.Type.INTEGER,
                "boolean": genai.protos.Type.BOOLEAN,
                "array": genai.protos.Type.ARRAY,
                "object": genai.protos.Type.OBJECT,
            }
            return type_mapping.get(python_type, genai.protos.Type.STRING)
        except Exception:
            return None


def get_llm_provider(provider_name: Optional[str] = None, **config) -> LLMProvider:
    """
    Factory function to get an LLM provider.
    
    Args:
        provider_name: Name of the provider ('openai', 'anthropic', 'ollama', 'gemini'). 
                      Defaults to LLM_PROVIDER environment variable or 'openai'.
        **config: Configuration for the provider (api_key, model, etc.)
    
    Returns:
        An LLMProvider instance
    
    Raises:
        ValueError: If the provider is not recognized
    """
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


def call_llm(prompt: str, provider_name: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
    """
    Convenience function to call an LLM.
    
    Args:
        prompt: The prompt to send to the LLM
        provider_name: The LLM provider to use. Defaults to environment variable or 'openai'.
        tools: Optional list of tool definitions in OpenAI format
        **kwargs: Additional arguments passed to the provider (model, api_key, etc.)
    
    Returns:
        The LLM's response as a string
    """
    provider = get_llm_provider(provider_name, **kwargs)
    return provider.call(prompt, tools=tools)
