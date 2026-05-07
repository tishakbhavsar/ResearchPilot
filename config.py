"""
config.py
The ONLY file you need to edit to configure ResearchPilot.

To use Grok (required by assignment):
    Set LLM_PROVIDER = "grok" and fill in GROK_API_KEY.

To use Gemini (alternative):
    Set LLM_PROVIDER = "gemini" and fill in GEMINI_API_KEY.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ── LLM Provider Selection ─────────────────────────────────────────────────────
LLM_PROVIDER = "gemini"

# ── Grok (xAI) ────────────────────────────────────────────────────────────────
# Get your key from: https://console.x.ai/
GROK_API_KEY = os.environ.get("GROK_API_KEY", "your-grok-api-key-here")
GROK_MODEL   = "grok-beta"

# ── Gemini (Google) ───────────────────────────────────────────────────────────
# Get your key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your-gemini-api-key-here")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
