#!/usr/bin/env python3
"""
Configuration file for ABM LLM simulation.
Contains API settings, proxy configurations, and other constants.
"""

import os

# Proxy settings
PROXY = None  # Set to your proxy if needed, e.g., {"http": "http://proxy:port", "https": "https://proxy:port"}

# Retry settings for LLM API calls
ATTEMPT_COUNTER = 3  # Number of retry attempts
WAIT_TIME_MIN = 1    # Minimum wait time between retries (seconds)
WAIT_TIME_MAX = 10   # Maximum wait time between retries (seconds)

# VLLM settings for local models
VLLM_URL = "http://localhost:8000/v1"  # Default VLLM server URL

# ZhipuAI settings
ZHIPUAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "max_input_tokens": 2000
} 