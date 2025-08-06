"""LLM configuration for Concordia integration.

Fill in your provider, model name and API key.  This file is imported by
`agents.fifteenminutescity.concordia_brain` so that all agents share the same
language-model instance.

Do NOT commit real API keys to version control – keep them in local overrides or
set them via environment variables.
"""

#import os

# OpenRouter configuration for DeepSeek access
#PROVIDER: str = "custom_openai"

# Model name for the chosen provider
#MODEL_NAME: str = "gpt-4.1-nano"
# DeepSeek model through OpenRouter
#MODEL_NAME: str = "deepseek/deepseek-r1:free"
# API key - preferably set via environment variable for security
# Set environment variable: OPENROUTER_API_KEY=sk-or-v1-your_actual_key_here
#API_KEY: str = os.getenv("OPENROUTER_API_KEY", "sk-nTB0IQ8qyGeHWpnB39FbA67b04F1402698D4FbB0Ad3fA30e")

# OpenRouter base URL
#BASE_URL: str | None = "https://openrouter.ai/api/v1" 


# API key (or leave blank and set the corresponding environment variable,
# e.g.  OPENAI_API_KEY, AZURE_OPENAI_KEY, etc.)
#API_KEY: str = "sk-Odug1h3sFBqLiAeC1c16E5B9028c4cBbA96e8b1003310519"

# Optional: custom base URL (useful for self-hosted or proxy endpoints)
#BASE_URL: str | None = "https://one-api.jiaancc.site/v1"  # Set to None to use default OpenAI URL 

# config/llm_config.py
import os

PROVIDER   : str = "custom_openai"      # tells ConcordiaBrain to use the wrapper that accepts a custom url
MODEL_NAME : str = "gpt-4.1-mini"            # the model string exposed by your proxy
API_KEY    : str = os.getenv("OPENROUTER_API_KEY",
                             "sk-nTB0IQ8qyGeHWpnB39FbA67b04F1402698D4FbB0Ad3fA30e")
BASE_URL   : str | None = "https://one-api.unu-macau.xyz/v1"   # ← **add this line**