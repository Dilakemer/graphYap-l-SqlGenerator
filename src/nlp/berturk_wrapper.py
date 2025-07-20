"""
BERTurk Wrapper with Local Cache Support
"""
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# Add config to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.model_config import (
    setup_model_environment,
    get_hf_token,
    BERTURK_MODEL_NAME,
    BERTURK_LOCAL_PATH
)

class BERTurkWrapper:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            setup_model_environment()
            self._load_model()
            self._initialized = True

    def _load_model(self):
        """Load BERTurk model with local-first strategy"""

        # Strategy 1: Load from local cache (preferred)
        if self._try_local_load():
            return

        # Strategy 2: Download and cache
        if self._download_and_load():
            return

        raise RuntimeError("Failed to load BERTurk model")

    def _try_local_load(self):
        """Try loading from local cache"""
        try:
            if (BERTURK_LOCAL_PATH / "config.json").exists():
                print("🔄 Loading BERTurk from local cache...")
                self._tokenizer = AutoTokenizer.from_pretrained(str(BERTURK_LOCAL_PATH))
                self._model = AutoModel.from_pretrained(str(BERTURK_LOCAL_PATH))
                self._model.eval()
                print("✅ Loaded from local cache")
                return True
        except Exception as e:
            print(f"⚠️ Local load failed: {e}")
        return False

    def _download_and_load(self):
        """Download model and load"""
        try:
            from huggingface_hub import login, snapshot_download

            token = get_hf_token()
            if not token:
                raise RuntimeError("No HF token available")

            print("🔑 Using HF token...")
            login(token=token)

            print("📥 Downloading BERTurk...")
            snapshot_download(
                repo_id=BERTURK_MODEL_NAME,
                local_dir=BERTURK_LOCAL_PATH,
                local_dir_use_symlinks=False,
                resume_download=True
            )

            # Load after download
            self._tokenizer = AutoTokenizer.from_pretrained(str(BERTURK_LOCAL_PATH))
            self._model = AutoModel.from_pretrained(str(BERTURK_LOCAL_PATH))
            self._model.eval()
            print("✅ Downloaded and loaded")
            return True

        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    def get_embeddings(self, text):
        """Get embeddings for single text"""
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                return embeddings.flatten().astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")

    def get_embeddings_batch(self, texts):
        """Get embeddings for multiple texts"""
        if not texts:
            raise ValueError("Text list cannot be empty")

        try:
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use [CLS] token embeddings
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                return embeddings.astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"Batch embedding generation failed: {e}")

    def get_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts"""
        embeddings = self.get_embeddings_batch([text1, text2])

        # Cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def is_loaded(self):
        """Check if model is loaded"""
        return hasattr(self, '_model') and self._model is not None

    def get_model_info(self):
        """Get model information"""
        return {
            "model_name": BERTURK_MODEL_NAME,
            "local_path": str(BERTURK_LOCAL_PATH),
            "model_loaded": self.is_loaded(),
            "max_length": 512,
            "embedding_dimension": 768
        }

# Convenience function
def get_berturk_instance():
    """Get BERTurk singleton instance"""
    return BERTurkWrapper()


