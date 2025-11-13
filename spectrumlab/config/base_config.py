import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root directory
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)


@dataclass
class Config:
    # DeepSeek API Configuration
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL")
    deepseek_model_name: str = os.getenv("DEEPSEEK_MODEL_NAME")

    # GPT-4o API Configuration
    gpt4o_api_key: str = os.getenv("GPT4O_API_KEY")
    gpt4o_base_url: str = os.getenv("GPT4O_BASE_URL")
    gpt4o_model_name: str = os.getenv("GPT4O_MODEL_NAME")

    # InternVL API Configuration
    internvl_api_key: str = os.getenv("INTERNVL_API_KEY")
    internvl_base_url: str = os.getenv("INTERNVL_BASE_URL")
    internvl_model_name: str = os.getenv("INTERNVL_MODEL_NAME")

    # Intern-S1 API Configuration
    intern_s1_api_key: str = os.getenv("INTERN_S1_API_KEY")
    intern_s1_base_url: str = os.getenv("INTERN_S1_BASE_URL")
    intern_s1_model_name: str = os.getenv("INTERN_S1_MODEL_NAME")

    # Intern-S1-Mini API Configuration
    intern_s1_mini_api_key: str = os.getenv("INTERN_S1_MINI_API_KEY")
    intern_s1_mini_base_url: str = os.getenv("INTERN_S1_MINI_BASE_URL")
    intern_s1_mini_model_name: str = os.getenv("INTERN_S1_MINI_MODEL_NAME")

    # Claude API Configuration
    claude_api_key: str = os.getenv("CLAUDE_API_KEY")
    claude_base_url: str = os.getenv("CLAUDE_BASE_URL")
    claude_sonnet_3_5_model_name: str = os.getenv("CLAUDE_SONNET_3_5")
    claude_opus_4_model_name: str = os.getenv("CLAUDE_OPUS_4")
    claude_haiku_3_5_model_name: str = os.getenv("CLAUDE_HAIKU_3_5")
    claude_sonnet_4_model_name: str = os.getenv("CLAUDE_SONNET_4")

    # GPT-4.1, GPT-4-Vision
    gpt4_1_api_key: str = os.getenv("GPT4_1_API_KEY")
    gpt4_1_base_url: str = os.getenv("GPT4_1_BASE_URL")
    gpt4_1_model_name: str = os.getenv("GPT4_1")
    gpt4_vision_api_key: str = os.getenv("GPT4_VISION_API_KEY")
    gpt4_vision_base_url: str = os.getenv("GPT4_VISION_BASE_URL")
    gpt4_vision_model_name: str = os.getenv("GPT4_VISION")

    # Grok-2-Vision
    grok_2_vision_api_key: str = os.getenv("GROK_2_VISION_API_KEY")
    grok_2_vision_base_url: str = os.getenv("GROK_2_VISION_BASE_URL")
    grok_2_vision_model_name: str = os.getenv("GROK_2_VISION")

    # Qwen-VL-Max
    qwen_vl_api_key: str = os.getenv("QWEN_VL_API_KEY")
    qwen_vl_base_url: str = os.getenv("QWEN_VL_BASE_URL")
    qwen_vl_model_name: str = os.getenv("QWEN_VL_MAX")

    # DeepSeek-VL-2
    deepseek_vl_2_api_key: str = os.getenv("DEEPSEEK_VL_2_API_KEY")
    deepseek_vl_2_base_url: str = os.getenv("DEEPSEEK_VL_2_BASE_URL")
    deepseek_vl_2_model_name: str = os.getenv("DEEPSEEK_VL_2")

    # Qwen-2.5-VL-32B
    qwen_2_5_vl_32b_api_key: str = os.getenv("QWEN_VL_API_KEY")
    qwen_2_5_vl_32b_base_url: str = os.getenv("QWEN_VL_BASE_URL")
    qwen_2_5_vl_32b_model_name: str = os.getenv("QWEN_2_5_VL_32B")

    # Qwen-2.5-VL-72B
    qwen_2_5_vl_72b_api_key: str = os.getenv("QWEN_VL_API_KEY")
    qwen_2_5_vl_72b_base_url: str = os.getenv("QWEN_VL_BASE_URL")
    qwen_2_5_vl_72b_model_name: str = os.getenv("QWEN_2_5_VL_72B")

    # Llama-Vision-11B
    llama_vision_11b_api_key: str = os.getenv("LLAMA_VISION_API_KEY")
    llama_vision_11b_base_url: str = os.getenv("LLAMA_VISION_BASE_URL")
    llama_vision_11b_model_name: str = os.getenv("LLAMA_VISION_11B")

    # Llama-Vision-90B
    llama_vision_90b_api_key: str = os.getenv("LLAMA_VISION_API_KEY")
    llama_vision_90b_base_url: str = os.getenv("LLAMA_VISION_BASE_URL")
    llama_vision_90b_model_name: str = os.getenv("LLAMA_VISION_90B")

    # Doubao-1.5-Vision-Pro
    doubao_1_5_vision_pro_api_key: str = os.getenv("DOUBAO_1_5_VISION_PRO_API_KEY")
    doubao_1_5_vision_pro_base_url: str = os.getenv("DOUBAO_1_5_VISION_PRO_BASE_URL")
    doubao_1_5_vision_pro_model_name: str = os.getenv("DOUBAO_1_5_VISION_PRO")

    # Doubao-1.5-Vision-Pro-Thinking
    doubao_1_5_vision_pro_thinking_api_key: str = os.getenv(
        "DOUBAO_1_5_VISION_PRO_THINKING_API_KEY"
    )
    doubao_1_5_vision_pro_thinking_base_url: str = os.getenv(
        "DOUBAO_1_5_VISION_PRO_THINKING_BASE_URL"
    )
    doubao_1_5_vision_pro_thinking_model_name: str = os.getenv(
        "DOUBAO_1_5_VISION_PRO_THINKING"
    )
