import os

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(__file__)

# --- Agent Configuration ---
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "8"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "5000"))
KEEP_MESSAGES = int(os.getenv("KEEP_MESSAGES", "20"))
