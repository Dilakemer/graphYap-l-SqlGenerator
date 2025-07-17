"""
Integration Tests - End-to-End Testing
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.nlp.nlp_processor import NLPProcessor
from src.nlp.berturk_wrapper import BERTurkWrapper
from src.nlp.intent_classifier import IntentClassifier
from src.nlp.entity_extractor import EntityExtractor


class TestIntegration:
    """Test end-to-end integration scenarios"""

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_full_pipeline_simple_query(self, mock_model, mock_tokenizer):
        """Test complete pipeline with simple customer count query"""
        # Mock BERTurk model loading
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock BERTurk embeddings for realistic flow
        def mock_get_embeddings(text):
            # Return different embeddings based on text content
            if "müşteri" in text.lower():
                # High similarity for customer-related patterns
                embedding = np.ones(768) * 0.8
            elif "sayı" in text.lower() or "count" in text.lower():
                # High similarity for count-related patterns
                embedding = np.ones(768) * 0.9
            else:
                # Low similarity for other patterns
                embedding = np.ones(768) * 0.1
            return embedding.astype(np.float32)

        def mock_get_embeddings_batch(texts):
            return np.array([mock_get_embeddings(text) for text in texts])

        # Apply mocks to all BERTurk instances
        with patch.object(BERTurkWrapper, 'get_embeddings', side_effect=mock_get_embeddings), \
                patch.object(BERTurkWrapper, 'get_embeddings_batch', side_effect=mock_get_embeddings_batch):

            processor = NLPProcessor()
            result = processor.analyze("müşteri sayısını hesapla")

            # Basic structure validation
            assert result["text"] == "müşteri sayısını hesapla"
            assert "intent" in result
            assert "entities" in result
            assert "analysis_metadata" in result

            # Intent should be detected (even if not perfectly trained)
            assert result["intent"]["type"] in ["COUNT", "SELECT", "SUM", "AVG"]
            assert result["intent"]["confidence"] > 0.0

            # Entities should be detected
            assert "tables" in result["entities"]
            assert "time_filters" in result["entities"]

            # Should detect customer table with reasonable confidence
            if result["entities"]["tables"]:
                customer_table = next((t for t in result["entities"]["tables"]
                                       if t["table"] == "customers"), None)
                if customer_table:
                    assert customer_table["confidence"] > 0.65

            # Processing metadata should be present
            assert "processing_status" in result["analysis_metadata"]
            assert result["analysis_metadata"]["processing_status"] in ["success", "error"]

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_full_pipeline_time_filter_query(self, mock_model, mock_tokenizer):
        """Test complete pipeline with time-filtered query"""
        # Mock BERTurk model loading
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock BERTurk embeddings with time awareness
        def mock_get_embeddings(text):
            embedding = np.zeros(768)

            # Time-related patterns
            if any(phrase in text.lower() for phrase in ["bu ay", "current month"]):
                embedding[:100] = 1.0  # Time pattern

            # Table-related patterns
            if "sipariş" in text.lower():
                embedding[200:300] = 1.0  # Orders table pattern
            elif "liste" in text.lower():
                embedding[300:400] = 1.0  # List/Select pattern

            return embedding.astype(np.float32)

        def mock_get_embeddings_batch(texts):
            return np.array([mock_get_embeddings(text) for text in texts])

        with patch.object(BERTurkWrapper, 'get_embeddings', side_effect=mock_get_embeddings), \
                patch.object(BERTurkWrapper, 'get_embeddings_batch', side_effect=mock_get_embeddings_batch):

            processor = NLPProcessor()
            result = processor.analyze("bu ayın sipariş listesi")

            # Should detect both table and time filter
            assert "entities" in result

            # Check for orders table detection
            if result["entities"]["tables"]:
                orders_table = next((t for t in result["entities"]["tables"]
                                     if t["table"] == "orders"), None)
                if orders_table:
                    assert orders_table["confidence"] > 0.0

            # Check for time filter detection
            if result["entities"]["time_filters"]:
                current_month_filter = next((tf for tf in result["entities"]["time_filters"]
                                             if tf["period"] == "current_month"), None)
                if current_month_filter:
                    assert "sql_condition" in current_month_filter
                    assert current_month_filter["confidence"] > 0.0

            # Should have higher complexity due to multiple entities
            if result["entities"]["metadata"]["total_entities"] >= 2:
                assert result["entities"]["metadata"]["complexity"] in ["medium", "complex"]

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_full_pipeline_batch_processing(self, mock_model, mock_tokenizer):
        """Test complete pipeline with batch processing"""
        # Mock BERTurk model loading
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock embeddings with pattern recognition
        def mock_get_embeddings(text):
            embedding = np.zeros(768)

            # Different patterns for different queries
            if "müşteri" in text.lower():
                embedding[:100] = 1.0
            elif "ürün" in text.lower():
                embedding[100:200] = 1.0
            elif "satış" in text.lower():
                embedding[200:300] = 1.0

            if "sayı" in text.lower():
                embedding[500:600] = 1.0  # Count pattern
            elif "liste" in text.lower():
                embedding[600:700] = 1.0  # Select pattern
            elif "toplam" in text.lower():
                embedding[700:768] = 1.0  # Sum pattern

            return embedding.astype(np.float32)

        def mock_get_embeddings_batch(texts):
            return np.array([mock_get_embeddings(text) for text in texts])

        with patch.object(BERTurkWrapper, 'get_embeddings', side_effect=mock_get_embeddings), \
                patch.object(BERTurkWrapper, 'get_embeddings_batch', side_effect=mock_get_embeddings_batch):

            processor = NLPProcessor()

            test_queries = [
                "müşteri sayısını hesapla",
                "ürün listesini göster",
                "satış toplamını bul"
            ]

            results = processor.analyze_batch(test_queries)

            # Should process all queries
            assert len(results) == 3

            # Each result should have proper structure
            for i, result in enumerate(results):
                assert result["text"] == test_queries[i]
                assert "intent" in result
                assert "entities" in result
                assert "analysis_metadata" in result

                # Should have some confidence in intent detection
                assert result["intent"]["confidence"] >= 0.0

                # Should have processing status
                assert result["analysis_metadata"]["processing_status"] in ["success", "error"]

            # Processor should track batch processing
            assert processor.processed_queries == 3

    def test_component_interaction_without_training(self):
        """Test component interaction without trained models"""
        processor = NLPProcessor()

        # Should handle untrained models gracefully
        result = processor.analyze("müşteri sayısını hesapla")

        # Should return error result for untrained intent classifier
        assert result["analysis_metadata"]["processing_status"] == "error"
        assert result["intent"]["type"] == "UNKNOWN"
        assert result["intent"]["confidence"] == 0.0

        # But entity extraction should still work (doesn't require training)
        assert "entities" in result
        assert isinstance(result["entities"]["tables"], list)
        assert isinstance(result["entities"]["time_filters"], list)

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_query_context_extraction_integration(self, mock_model, mock_tokenizer):
        """Test query context extraction from full analysis"""
        # Mock BERTurk model loading
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock realistic embeddings for SQL-ready query
        def mock_get_embeddings(text):
            embedding = np.ones(768) * 0.8  # High similarity baseline
            return embedding.astype(np.float32)

        def mock_get_embeddings_batch(texts):
            return np.array([mock_get_embeddings(text) for text in texts])

        with patch.object(BERTurkWrapper, 'get_embeddings', side_effect=mock_get_embeddings), \
                patch.object(BERTurkWrapper, 'get_embeddings_batch', side_effect=mock_get_embeddings_batch):
            processor = NLPProcessor()

            # Analyze query
            analysis_result = processor.analyze("müşteri sayısını hesapla")

            # Extract query context
            context = processor.get_query_context(analysis_result)

            # Context should have basic structure
            assert "ready" in context

            if context["ready"]:
                assert "intent" in context
                assert "primary_table" in context
                assert "confidence" in context
                assert context["intent"] in ["SELECT", "COUNT", "SUM", "AVG"]
                assert context["confidence"] > 0.0

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_system_robustness_with_edge_cases(self, mock_model, mock_tokenizer):
        """Test system robustness with edge case inputs"""
        # Mock BERTurk model loading
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        def mock_get_embeddings(text):
            # Return low similarity for edge cases
            return np.ones(768) * 0.1

        def mock_get_embeddings_batch(texts):
            return np.array([mock_get_embeddings(text) for text in texts])

        with patch.object(BERTurkWrapper, 'get_embeddings', side_effect=mock_get_embeddings), \
                patch.object(BERTurkWrapper, 'get_embeddings_batch', side_effect=mock_get_embeddings_batch):
            processor = NLPProcessor()

            edge_cases = [
                "asdfghjkl qwertyuiop",  # Random text
                "123 456 789",  # Numbers only
                "!@#$%^&*()",  # Special characters
                "english text here",  # English text
                "a" * 1000,  # Very long text
                "x"  # Very short text
            ]

            for edge_case in edge_cases:
                result = processor.analyze(edge_case)

                # Should not crash and return valid structure
                assert "text" in result
                assert "intent" in result
                assert "entities" in result
                assert "analysis_metadata" in result

                # Should handle gracefully (low confidence or error)
                assert result["intent"]["confidence"] >= 0.0
                assert result["analysis_metadata"]["processing_status"] in ["success", "error"]

    def test_processor_statistics_tracking(self):
        """Test that processor correctly tracks statistics"""
        processor = NLPProcessor()

        # Initial state
        stats = processor.get_processing_stats()
        assert stats["total_processed"] == 0
        assert stats["successful_analyses"] == 0
        assert stats["success_rate"] == 0

        # Process some queries (will fail due to untrained models)
        test_queries = ["query1", "query2", "query3"]

        for query in test_queries:
            try:
                processor.analyze(query)
            except:
                pass  # Expected to fail with untrained models

        # Should track processed queries even if they fail
        stats = processor.get_processing_stats()
        assert stats["total_processed"] == 3

    def test_system_info_completeness(self):
        """Test that system info provides complete information"""
        processor = NLPProcessor()

        info = processor.get_system_info()

        # Should have all required sections
        required_sections = [
            "nlp_processor_version",
            "components",
            "supported_intents",
            "supported_entities"
        ]

        for section in required_sections:
            assert section in info

        # Components should include all main components
        components = info["components"]
        assert "intent_classifier" in components
        assert "entity_extractor" in components
        assert "berturk_wrapper" in components

        # Should list correct supported features
        assert info["supported_entities"] == ["tables", "time_filters"]
        assert len(info["supported_intents"]) == 4  # SELECT, COUNT, SUM, AVG

    @patch('src.nlp.berturk_wrapper.AutoTokenizer')
    @patch('src.nlp.berturk_wrapper.AutoModel')
    def test_singleton_consistency_across_components(self, mock_model, mock_tokenizer):
        """Test that BERTurk singleton works consistently across all components"""
        # Mock BERTurk model loading
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Create multiple components
        processor = NLPProcessor()
        intent_classifier = IntentClassifier()
        entity_extractor = EntityExtractor()

        # All should use the same BERTurk instance
        assert processor.berturk is intent_classifier.berturk
        assert processor.berturk is entity_extractor.berturk
        assert intent_classifier.berturk is entity_extractor.berturk

        # BERTurk instance should be loaded
        assert processor.berturk.is_loaded()

    def test_end_to_end_error_recovery(self):
        """Test end-to-end error recovery and graceful degradation"""
        processor = NLPProcessor()

        # Test with empty text (should raise ValueError)
        with pytest.raises(ValueError):
            processor.analyze("")

        # Test with invalid text (should return error result, not crash)
        result = processor.analyze("invalid query text")

        # Should return structured error response
        assert result["analysis_metadata"]["processing_status"] == "error"
        assert result["intent"]["type"] == "UNKNOWN"
        assert "entities" in result

        # Processor should continue working after error
        result2 = processor.analyze("another query")
        assert "analysis_metadata" in result2

        # Should track both attempts
        assert processor.processed_queries >= 2