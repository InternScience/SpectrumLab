from typing import Optional, Union, Dict, Any
from .base_api import BaseAPIModel
from spectrumlab.config import Config
from openai import OpenAI


class Intern_S1(BaseAPIModel):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        thinking_mode: bool = True,
        **kwargs,
    ):
        config = Config()

        # Use provided parameters or fall back to config
        self.api_key = api_key or config.intern_s1_api_key
        self.base_url = base_url or config.intern_s1_base_url
        self.model_name = model_name or config.intern_s1_model_name
        self.thinking_mode = thinking_mode

        # Validate that we have required configuration
        if not self.api_key:
            raise ValueError(
                "Intern-S1 API key not found. Please set INTERN_S1_API_KEY in your .env file "
                "or provide api_key parameter."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Initialize parent class
        super().__init__(model_name=self.model_name, **kwargs)

    def generate(
        self,
        prompt: Union[str, Dict[str, Any]],
        max_tokens: int = 512,
        **generation_kwargs,
    ) -> str:
        """
        Generate response supporting both text and multimodal input.

        Args:
            prompt: Either text string or multimodal dict
            max_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters like temperature, top_p, etc.
                                Can include extra_body to override thinking_mode.

        Returns:
            Generated response string. For Intern-S1, the response is always retrieved from
            reasoning_content field as the API returns empty content field.

        Note:
            Intern-S1 API behavior:
            - The content field is always empty
            - The actual response is always in reasoning_content field
            - This is true regardless of thinking_mode parameter
            - thinking_mode is set during model initialization, not in generate()
        """
        messages = []

        if isinstance(prompt, dict) and "images" in prompt:
            content = []

            content.append({"type": "text", "text": prompt["text"]})

            for image_data in prompt["images"]:
                content.append(image_data)

            messages.append({"role": "user", "content": content})
        else:
            text_content = prompt if isinstance(prompt, str) else prompt.get("text", "")
            messages.append({"role": "user", "content": text_content})

        # Prepare API call parameters
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # Handle thinking_mode and extra_body
        # If extra_body is provided in generation_kwargs, use it; otherwise use instance thinking_mode
        if "extra_body" not in generation_kwargs and self.thinking_mode:
            api_params["extra_body"] = {"thinking_mode": True}

        # Add other generation parameters
        for key, value in generation_kwargs.items():
            if key != "extra_body" or "extra_body" not in api_params:
                api_params[key] = value

        try:
            response = self.client.chat.completions.create(**api_params)
            message = response.choices[0].message

            # Intern-S1 API always returns content in reasoning_content field
            # The content field is always empty, regardless of thinking_mode
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                return message.reasoning_content
            # Fallback to regular content (should not happen for Intern-S1)
            return message.content or ""
        except Exception as e:
            raise RuntimeError(f"Intern-S1 API call failed: {e}")


class Intern_S1_Mini(BaseAPIModel):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        config = Config()

        # Use provided parameters or fall back to config
        self.api_key = api_key or config.intern_s1_mini_api_key
        self.base_url = base_url or config.intern_s1_mini_base_url
        self.model_name = model_name or config.intern_s1_mini_model_name

        # Validate that we have required configuration
        if not self.api_key:
            raise ValueError(
                "Intern-S1-Mini API key not found. Please set INTERN_S1_MINI_API_KEY in your .env file "
                "or provide api_key parameter."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Initialize parent class
        super().__init__(model_name=self.model_name, **kwargs)

    def generate(
        self,
        prompt: Union[str, Dict[str, Any]],
        max_tokens: int = 512,
        **generation_kwargs,
    ) -> str:
        """
        Generate response supporting both text and multimodal input.

        Args:
            prompt: Either text string or multimodal dict
            max_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters like temperature, top_p, etc.

        Returns:
            Generated response string. For Intern-S1-Mini, the response is always retrieved from
            reasoning_content field as the API returns empty content field.

        Note:
            Intern-S1-Mini API behavior:
            - The content field is always empty
            - The actual response is always in reasoning_content field
            - This behavior is consistent with Intern-S1
        """
        messages = []

        if isinstance(prompt, dict) and "images" in prompt:
            content = []

            content.append({"type": "text", "text": prompt["text"]})

            for image_data in prompt["images"]:
                content.append(image_data)

            messages.append({"role": "user", "content": content})
        else:
            text_content = prompt if isinstance(prompt, str) else prompt.get("text", "")
            messages.append({"role": "user", "content": text_content})

        # Prepare API call parameters
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # Add generation parameters
        for key, value in generation_kwargs.items():
            api_params[key] = value

        try:
            response = self.client.chat.completions.create(**api_params)
            message = response.choices[0].message

            # Intern-S1-Mini API always returns content in reasoning_content field
            # The content field is always empty, similar to Intern-S1
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                return message.reasoning_content
            # Fallback to regular content (should not happen for Intern-S1-Mini)
            return message.content or ""
        except Exception as e:
            raise RuntimeError(f"Intern-S1-Mini API call failed: {e}")
