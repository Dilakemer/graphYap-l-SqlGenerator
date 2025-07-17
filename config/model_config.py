"""
BERTurk model configuration and Setup
"""

import os
from pathlib import Path

# Model paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BERTURK_LOCAL_PATH = MODELS_DIR / "turkish-bert"
CACHE_DIR = MODELS_DIR / "cache"


# Model identifiers
BERTURK_MODEL_NAME = "dbmdz/bert-base-turkish-cased"


# Environment setup
def setup_model_environment():
    """Setup model directories and environment"""
    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)
    BERTURK_LOCAL_PATH.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    # Set environment variables for caching
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
    os.environ["HF_HOME"] = str(CACHE_DIR)

    return {
        "models_dir": MODELS_DIR,
        "berturk_path": BERTURK_LOCAL_PATH,
        "cache_dir": CACHE_DIR
    }


def get_hf_token():
    """Get HF token from environment or file"""
    # Try environment first
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    # Try token file
    token_file = PROJECT_ROOT / ".hf_token"
    if token_file.exists():
        return token_file.read_text().strip()

    return None
