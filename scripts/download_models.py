#!/usr/bin/env python3
"""
Model Download Script - One-time setup
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import (
    setup_model_environment,
    get_hf_token,
    BERTURK_MODEL_NAME,
    BERTURK_LOCAL_PATH
)

from huggingface_hub import login, snapshot_download


def download_berturk():
    """Download BERTurk model for offline use"""

    # Setup environment
    paths = setup_model_environment()
    print(f"📁 Models directory: {paths['models_dir']}")

    # Get token
    token = get_hf_token()
    if not token:
        print("❌ No HF token found!")
        print("   Set HF_TOKEN environment variable or create .hf_token file")
        return False

    print("🔑 HF Token found")

    # Login
    try:
        login(token=token)
        print("✅ Logged in to Hugging Face")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

    # Check if already downloaded
    if (BERTURK_LOCAL_PATH / "config.json").exists():
        print("✅ BERTurk already downloaded locally")
        return True

    # Download
    print("📥 Downloading BERTurk model...")
    try:
        snapshot_download(
            repo_id=BERTURK_MODEL_NAME,
            local_dir=BERTURK_LOCAL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"✅ BERTurk downloaded to: {BERTURK_LOCAL_PATH}")
        print("✅ Model ready for offline use!")
        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def verify_download():
    """Verify downloaded model"""
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.txt"
    ]

    for file in required_files:
        file_path = BERTURK_LOCAL_PATH / file
        if not file_path.exists():
            print(f"❌ Missing: {file}")
            return False

    print("✅ All model files present")
    return True


if __name__ == "__main__":
    print("🚀 BERTurk Model Download Script")
    print("=" * 40)

    success = download_berturk()
    if success:
        verify_download()

    print("=" * 40)
    print("Done!")