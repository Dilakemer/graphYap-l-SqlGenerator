"""
BERTurk Wrapper - Türkçe destekli doğal dil işleme için temel
Singleton pattern ile memory-efficient BERTurk model management.
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class BERTurkWrapper:
    """
    Singleton BERTurk wrapper for turkish text processing.
    Provides cached model loading and efficient embedding extraction
    """

    _instance = None
    _model = None
    _tokenizer = None
    _model_loaded = False

    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(BERTurkWrapper, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize wrapper - model loaded on first use"""
        if not hasattr(self, "initialized"):
            self.model_name = "dbmdz/bert-base-turkish-cased"
            self.max_length = 512
            self.initialized = True

            # Load model on initialization
            self._load_model()

    def _load_model(self):
        """Load BERTurk model and tokenizer with error handling"""
        if self._model_loaded:
            return

        try:
            # Load tokenizer and the model
            self._tokenizer = AutoTokenizer.from_pretrained("self.model_name")
            self._model = AutoModel.from_pretrained(self.model_name)

            # Set to evaluation mode
            self._model.eval()

            self._model_loaded = True

        except Exception as e:
            raise RuntimeError(f"BERTurk model loading failed: {e}")

    def get_embeddings(self, text):
        """
        Extract emdeddings from turkish text

        Args:
            text: Turkish text input
        Returns:
            numpy array of shape (768,) containing text embeddings
        """

        if not self._model_loaded:
            raise RuntimeError("BERTurk model not loaded")

        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        try:
            # Tokenize text
            inputs = self._tokenizer(
                text,
                # we want the output to be same as pytorch tensors not tensorflow tensors etc.
                return_tensors="pt",
                # Make the sentences equal length in order for the self-attention mechanism work correctly
                padding=True,
                # If the lenght is more than 512, then truncate after the 512.
                truncation=True,
                # Accept an array of tokens whose length is 512 or less
                max_length=self.max_length,
            )

            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)

                # Extract CLS token embedding (global represantation)
                # [CLS] represents the general meaning of the sentence
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
                # : -> all batches, 0 -> first token which is CLS, : -> whole embedding dimensions which is 768
                # .numpy() -> pytorch tensor is transformed into  numpy array

            # flatten the array so that it is in the form of which sklearn can accept
            return cls_embedding.flatten()

        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")

    def get_embeddings_batch(self, texts):
        """
        Extract embeddings for multiple texts (batch processing)

        Args:
            texts: List of Turkish text inputs

        Returns:
            numpy array of shape (n, 768) containing embeddings for all texts
        """

        if not texts:
            raise ValueError("Text list cannot be empty.")

        if not self._model_loaded:
            raise RuntimeError("BERTurk model not downloaded")

        try:
            # Tokenize all texts
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)

                # Extract CLS token embeddings
                cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

            return cls_embeddings

        except Exception as e:
            raise RuntimeError(f"Batch embedding generation failed: {e}")

    def get_similarity(self, text1, text2):
        """
        Calculate cosine similarity between two turkish texts

        Args:
            text1: First turkish text
            text2: Second turkish text

        Returns:
            Cosine similarity score between 0 and 1
        """

        try:
            embeddings = self.get_embeddings_batch([text1, text2])

            # Calculate the cosine similarity
            embedding1 = embeddings[0]
            embedding2 = embeddings[1]

            # Cosine similarity formula
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            similarity = dot_product / (norm1 * norm2)

            return float(similarity)

        except Exception as e:
            return 0.0

    def is_loaded(self):
        """Check if model is loaded"""
        return self._model_loaded

    def get_model_info(self):
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_loaded": self._model_loaded,
            "max_length": self.max_length,
            "embedding_dimension": 768,
        }


def get_berturk_instance():
    return BERTurkWrapper()
