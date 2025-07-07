"""LLM configuration for Concordia integration.

Fill in your provider, model name and API key.  This file is imported by
`agents.fifteenminutescity.concordia_brain` so that all agents share the same
language-model instance.

Do NOT commit real API keys to version control – keep them in local overrides or
set them via environment variables.
"""

# Which provider wrapper from `ref_Concordia_repo/concordia/language_model` to
# instantiate. Supported examples: "openai", "azure_openai", "ollama", etc.
PROVIDER: str = "custom_openai"

# Model name for the chosen provider
MODEL_NAME: str = "gpt-4.1-nano"

# API key (or leave blank and set the corresponding environment variable,
# e.g.  OPENAI_API_KEY, AZURE_OPENAI_KEY, etc.)
API_KEY: str = "sk-0DBN4qc3xmE4TukADf241a1eCeA94a6f8c221d8646848669"

# Optional: custom base URL (useful for self-hosted or proxy endpoints)
BASE_URL: str | None = "https://one-api.jiaancc.site/v1"  # Set to None to use default OpenAI URL 