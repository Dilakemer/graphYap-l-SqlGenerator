"""
BERTurk Wrapper Tests
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.nlp.berturk_wrapper import BERTurkWrapper, get_berturk_instance


class TestBERTurkWrapper:
    """Test BERTurk wrapper functionality"""

    def setup_method(self):
        """Reset singleton before each test"""
        BERTurkWrapper._instance = None
        BERTurkWrapper._model_loaded = False

    def test_singleton_pattern(self):
        """Test singleton pattern works correctly"""
        wrapper1 = BERTurkWrapper()
        wrapper2 = BERTurkWrapper()

        assert wrapper1 is wrapper2
        assert id(wrapper1) == id(wrapper2)

    def test_get_berturk_instance(self):
        """Test convenience function"""
        instance = get_berturk_instance()

        assert isinstance(instance, BERTurkWrapper)
        # Should be same instance due to singleton
        instance2 = get_berturk_instance()
        assert instance is instance2

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_model_loading(self, mock_model, mock_tokenizer):
        """Test model loading process"""
        # Mock tokenizer and model
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        wrapper = BERTurkWrapper()

        # Verify model loading was called
        mock_tokenizer.from_pretrained.assert_called_once_with("dbmdz/bert-base-turkish-cased")
        mock_model.from_pretrained.assert_called_once_with("dbmdz/bert-base-turkish-cased")
        mock_model_instance.eval.assert_called_once()

        assert wrapper.is_loaded()

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_get_embeddings_shape(self, mock_model, mock_tokenizer):
        """Test embedding output shape"""
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.return_value = {
            'input_ids': MagicMock(),
            'attention_mask': MagicMock()
        }

        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = np.random.rand(1, 10, 768)
        mock_model_instance.return_value = mock_output

        wrapper = BERTurkWrapper()
        embedding = wrapper.get_embeddings("test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_get_embeddings_batch_shape(self, mock_model, mock_tokenizer):
        """Test batch embedding output shape"""
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.return_value = {
            'input_ids': MagicMock(),
            'attention_mask': MagicMock()
        }

        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock model output for batch
        mock_output = MagicMock()
        mock_output.last_hidden_state = np.random.rand(3, 10, 768)
        mock_model_instance.return_value = mock_output

        wrapper = BERTurkWrapper()
        texts = ["text1", "text2", "text3"]
        embeddings = wrapper.get_embeddings_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 768)

    def test_get_embeddings_empty_text(self):
        """Test error handling for empty text"""
        wrapper = BERTurkWrapper()

        with pytest.raises(ValueError, match="Text input cannot be empty"):
            wrapper.get_embeddings("")

        with pytest.raises(ValueError, match="Text input cannot be empty"):
            wrapper.get_embeddings("   ")

    def test_get_embeddings_batch_empty_list(self):
        """Test error handling for empty list"""
        wrapper = BERTurkWrapper()

        with pytest.raises(ValueError, match="Text list cannot be empty"):
            wrapper.get_embeddings_batch([])

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_get_similarity(self, mock_model, mock_tokenizer):
        """Test similarity calculation"""
        # Mock components
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock get_embeddings_batch to return controlled embeddings
        wrapper = BERTurkWrapper()

        # Create test embeddings
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([1.0, 0.0, 0.0])  # Same as embedding1
        embedding3 = np.array([0.0, 1.0, 0.0])  # Different from embedding1

        with patch.object(wrapper, 'get_embeddings_batch') as mock_batch:
            # Test high similarity
            mock_batch.return_value = np.array([embedding1, embedding2])
            similarity_high = wrapper.get_similarity("text1", "text2")

            # Test low similarity
            mock_batch.return_value = np.array([embedding1, embedding3])
            similarity_low = wrapper.get_similarity("text1", "text3")

            assert 0.0 <= similarity_high <= 1.0
            assert 0.0 <= similarity_low <= 1.0
            assert similarity_high > similarity_low

    def test_get_model_info(self):
        """Test model information"""
        wrapper = BERTurkWrapper()
        info = wrapper.get_model_info()

        assert info["model_name"] == "dbmdz/bert-base-turkish-cased"
        assert info["max_length"] == 512
        assert info["embedding_dimension"] == 768
        assert "model_loaded" in info
        assert isinstance(info["model_loaded"], bool)

    def test_is_loaded(self):
        """Test model loading status"""
        wrapper = BERTurkWrapper()

        # Should return boolean
        status = wrapper.is_loaded()
        assert isinstance(status, bool)

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_model_loading_error(self, mock_model, mock_tokenizer):
        """Test model loading error handling"""
        # Mock tokenizer to raise exception
        mock_tokenizer.from_pretrained.side_effect = Exception("Model loading failed")

        with pytest.raises(RuntimeError, match="BERTurk model loading failed"):
            BERTurkWrapper()

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_embedding_generation_error(self, mock_model, mock_tokenizer):
        """Test embedding generation error handling"""
        # Mock successful model loading
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        wrapper = BERTurkWrapper()

        # Mock tokenizer call to raise exception
        mock_tokenizer.from_pretrained.return_value.side_effect = Exception("Tokenization failed")

        with pytest.raises(RuntimeError, match="Embedding generation failed"):
            wrapper.get_embeddings("test text")

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_model_eval_mode(self, mock_model, mock_tokenizer):
        """Test that model is set to evaluation mode"""
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        BERTurkWrapper()

        # Verify eval() was called
        mock_model_instance.eval.assert_called_once()

    def test_similarity_edge_cases(self):
        """Test similarity calculation edge cases"""
        wrapper = BERTurkWrapper()

        # Test with zero vectors
        with patch.object(wrapper, 'get_embeddings_batch') as mock_batch:
            # Zero vectors should return 0 similarity
            zero_embedding = np.zeros(768)
            mock_batch.return_value = np.array([zero_embedding, zero_embedding])

            similarity = wrapper.get_similarity("text1", "text2")
            assert similarity == 0.0

    def test_multiple_instantiation(self):
        """Test multiple instantiation returns same object"""
        instances = [BERTurkWrapper() for _ in range(5)]

        # All should be the same instance
        first_instance = instances[0]
        for instance in instances[1:]:
            assert instance is first_instance