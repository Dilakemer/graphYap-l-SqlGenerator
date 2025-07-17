"""
NLP Processor Tests
"""
import pytest
from unittest.mock import patch, MagicMock
from src.nlp.nlp_processor import NLPProcessor, create_nlp_processor


class TestNLPProcessor:
    """Test NLP processor functionality"""

    def test_processor_initialization(self):
        """Test processor initializes correctly"""
        processor = NLPProcessor()

        assert processor.processed_queries == 0
        assert processor.successful_analyses == 0
        assert hasattr(processor, 'intent_classifier')
        assert hasattr(processor, 'entity_extractor')
        assert hasattr(processor, 'berturk')

    def test_create_nlp_processor(self):
        """Test convenience function"""
        processor = create_nlp_processor()

        assert isinstance(processor, NLPProcessor)

    def test_analyze_empty_text(self):
        """Test analysis with empty text"""
        processor = NLPProcessor()

        with pytest.raises(ValueError, match="Text input cannot be empty"):
            processor.analyze("")

        with pytest.raises(ValueError, match="Text input cannot be empty"):
            processor.analyze("   ")

    @patch('src.nlp.nlp_processor.IntentClassifier')
    @patch('src.nlp.nlp_processor.EntityExtractor')
    @patch('src.nlp.nlp_processor.BERTurkWrapper')
    def test_analyze_success(self, mock_berturk, mock_entity_extractor, mock_intent_classifier):
        """Test successful analysis"""
        # Mock intent classifier
        mock_intent_instance = MagicMock()
        mock_intent_classifier.return_value = mock_intent_instance
        mock_intent_instance.predict.return_value = {
            "intent": "COUNT",
            "confidence": 0.85,
            "all_confidences": {
                "COUNT": 0.85,
                "SELECT": 0.10,
                "SUM": 0.03,
                "AVG": 0.02
            },
            "text": "müşteri sayısını hesapla"
        }

        # Mock entity extractor
        mock_entity_instance = MagicMock()
        mock_entity_extractor.return_value = mock_entity_instance
        mock_entity_instance.extract.return_value = {
            "tables": [{"table": "customers", "confidence": 0.9}],
            "time_filters": [],
            "metadata": {
                "total_entities": 1,
                "has_time_filter": False,
                "requires_join": False,
                "complexity": "simple"
            }
        }

        # Mock BERTurk
        mock_berturk_instance = MagicMock()
        mock_berturk.return_value = mock_berturk_instance

        processor = NLPProcessor()
        result = processor.analyze("müşteri sayısını hesapla")

        # Check result structure
        assert "text" in result
        assert "intent" in result
        assert "entities" in result
        assert "analysis_metadata" in result

        # Check intent structure
        assert result["intent"]["type"] == "COUNT"
        assert result["intent"]["confidence"] == 0.85
        assert "all_confidences" in result["intent"]

        # Check entities structure
        assert "tables" in result["entities"]
        assert "time_filters" in result["entities"]
        assert "metadata" in result["entities"]

        # Check metadata
        assert result["analysis_metadata"]["processing_status"] == "success"
        assert result["analysis_metadata"]["sql_ready"] == True
        assert processor.processed_queries == 1
        assert processor.successful_analyses == 1

    @patch('src.nlp.nlp_processor.IntentClassifier')
    @patch('src.nlp.nlp_processor.EntityExtractor')
    @patch('src.nlp.nlp_processor.BERTurkWrapper')
    def test_analyze_with_time_filter(self, mock_berturk, mock_entity_extractor, mock_intent_classifier):
        """Test analysis with time filter"""
        # Mock intent classifier
        mock_intent_instance = MagicMock()
        mock_intent_classifier.return_value = mock_intent_instance
        mock_intent_instance.predict.return_value = {
            "intent": "SELECT",
            "confidence": 0.88,
            "all_confidences": {"SELECT": 0.88, "COUNT": 0.12},
            "text": "bu ayın sipariş listesi"
        }

        # Mock entity extractor with time filter
        mock_entity_instance = MagicMock()
        mock_entity_extractor.return_value = mock_entity_instance
        mock_entity_instance.extract.return_value = {
            "tables": [{"table": "orders", "confidence": 0.85}],
            "time_filters": [
                {
                    "period": "current_month",
                    "confidence": 0.88,
                    "sql_condition": "DATE_TRUNC('month', CURRENT_DATE)"
                }
            ],
            "metadata": {
                "total_entities": 2,
                "has_time_filter": True,
                "requires_join": False,
                "complexity": "medium"
            }
        }

        processor = NLPProcessor()
        result = processor.analyze("bu ayın sipariş listesi")

        # Check time filter detection
        assert result["analysis_metadata"]["has_time_filter"] == True
        assert len(result["entities"]["time_filters"]) == 1
        assert result["entities"]["time_filters"][0]["period"] == "current_month"
        assert result["analysis_metadata"]["entity_complexity"] == "medium"

    @patch('src.nlp.nlp_processor.IntentClassifier')
    @patch('src.nlp.nlp_processor.EntityExtractor')
    def test_analyze_error_handling(self, mock_entity_extractor, mock_intent_classifier):
        """Test analysis error handling"""
        # Mock intent classifier to raise exception
        mock_intent_instance = MagicMock()
        mock_intent_classifier.return_value = mock_intent_instance
        mock_intent_instance.predict.side_effect = Exception("Intent classification failed")

        processor = NLPProcessor()
        result = processor.analyze("test text")

        # Check error handling
        assert result["analysis_metadata"]["processing_status"] == "error"
        assert "error_message" in result["analysis_metadata"]
        assert result["analysis_metadata"]["sql_ready"] == False
        assert result["intent"]["type"] == "UNKNOWN"
        assert result["intent"]["confidence"] == 0.0

    def test_analyze_batch_empty_list(self):
        """Test batch analysis with empty list"""
        processor = NLPProcessor()

        with pytest.raises(ValueError, match="Text list cannot be empty"):
            processor.analyze_batch([])

    @patch('src.nlp.nlp_processor.IntentClassifier')
    @patch('src.nlp.nlp_processor.EntityExtractor')
    @patch('src.nlp.nlp_processor.BERTurkWrapper')
    def test_analyze_batch_success(self, mock_berturk, mock_entity_extractor, mock_intent_classifier):
        """Test successful batch analysis"""
        # Mock intent classifier
        mock_intent_instance = MagicMock()
        mock_intent_classifier.return_value = mock_intent_instance
        mock_intent_instance.predict.side_effect = [
            {"intent": "COUNT", "confidence": 0.85, "all_confidences": {"COUNT": 0.85}},
            {"intent": "SELECT", "confidence": 0.90, "all_confidences": {"SELECT": 0.90}}
        ]

        # Mock entity extractor
        mock_entity_instance = MagicMock()
        mock_entity_extractor.return_value = mock_entity_instance
        mock_entity_instance.extract.side_effect = [
            {
                "tables": [{"table": "customers"}],
                "time_filters": [],
                "metadata": {"complexity": "simple", "requires_join": False, "has_time_filter": False}
            },
            {
                "tables": [{"table": "products"}],
                "time_filters": [],
                "metadata": {"complexity": "simple", "requires_join": False, "has_time_filter": False}
            }
        ]

        processor = NLPProcessor()
        texts = ["müşteri sayısı", "ürün listesi"]
        results = processor.analyze_batch(texts)

        assert len(results) == 2
        assert results[0]["intent"]["type"] == "COUNT"
        assert results[1]["intent"]["type"] == "SELECT"
        assert processor.processed_queries == 2

    @patch('src.nlp.nlp_processor.IntentClassifier')
    @patch('src.nlp.nlp_processor.EntityExtractor')
    def test_analyze_batch_with_error(self, mock_entity_extractor, mock_intent_classifier):
        """Test batch analysis with some errors"""
        # Mock intent classifier - first succeeds, second fails
        mock_intent_instance = MagicMock()
        mock_intent_classifier.return_value = mock_intent_instance
        mock_intent_instance.predict.side_effect = [
            {"intent": "COUNT", "confidence": 0.85, "all_confidences": {"COUNT": 0.85}},
            Exception("Failed")
        ]

        # Mock entity extractor
        mock_entity_instance = MagicMock()
        mock_entity_extractor.return_value = mock_entity_instance
        mock_entity_instance.extract.return_value = {
            "tables": [{"table": "customers"}],
            "time_filters": [],
            "metadata": {"complexity": "simple"}
        }

        processor = NLPProcessor()
        texts = ["müşteri sayısı", "invalid text"]
        results = processor.analyze_batch(texts)

        assert len(results) == 2
        assert results[0]["analysis_metadata"]["processing_status"] == "success"
        assert results[1]["analysis_metadata"]["processing_status"] == "error"
        assert results[1]["intent"]["type"] == "ERROR"

    def test_is_sql_ready_success(self):
        """Test SQL readiness check - success case"""
        processor = NLPProcessor()

        intent_result = {"intent": "COUNT", "confidence": 0.85}
        entity_result = {"tables": [{"table": "customers"}]}

        result = processor._is_sql_ready(intent_result, entity_result)
        assert result == True

    def test_is_sql_ready_low_confidence(self):
        """Test SQL readiness check - low confidence"""
        processor = NLPProcessor()

        intent_result = {"intent": "COUNT", "confidence": 0.5}  # Below threshold
        entity_result = {"tables": [{"table": "customers"}]}

        result = processor._is_sql_ready(intent_result, entity_result)
        assert result == False

    def test_is_sql_ready_no_tables(self):
        """Test SQL readiness check - no tables"""
        processor = NLPProcessor()

        intent_result = {"intent": "COUNT", "confidence": 0.85}
        entity_result = {"tables": []}  # No tables

        result = processor._is_sql_ready(intent_result, entity_result)
        assert result == False

    def test_is_sql_ready_unsupported_intent(self):
        """Test SQL readiness check - unsupported intent"""
        processor = NLPProcessor()

        intent_result = {"intent": "UNKNOWN", "confidence": 0.85}
        entity_result = {"tables": [{"table": "customers"}]}

        result = processor._is_sql_ready(intent_result, entity_result)
        assert result == False

    def test_get_query_context_not_ready(self):
        """Test query context when analysis not ready"""
        processor = NLPProcessor()

        analysis_result = {
            "analysis_metadata": {"sql_ready": False}
        }

        context = processor.get_query_context(analysis_result)

        assert context["ready"] == False
        assert "reason" in context

    def test_get_query_context_success(self):
        """Test query context extraction - success case"""
        processor = NLPProcessor()

        analysis_result = {
            "analysis_metadata": {"sql_ready": True, "requires_join": False},
            "intent": {"type": "COUNT", "confidence": 0.85},
            "entities": {
                "tables": [{"table": "customers", "confidence": 0.9}],
                "time_filters": [
                    {
                        "period": "current_month",
                        "sql_condition": "DATE_TRUNC('month', CURRENT_DATE)"
                    }
                ]
            }
        }

        context = processor.get_query_context(analysis_result)

        assert context["ready"] == True
        assert context["intent"] == "COUNT"
        assert context["primary_table"] == "customers"
        assert context["time_filter"]["period"] == "current_month"
        assert context["requires_join"] == False
        assert context["confidence"] == 0.85

    def test_validate_analysis_success(self):
        """Test analysis validation - success case"""
        processor = NLPProcessor()

        analysis_result = {
            "intent": {"confidence": 0.85},
            "entities": {
                "tables": [{"table": "customers"}],
                "metadata": {"complexity": "simple"}
            },
            "analysis_metadata": {"sql_ready": True}
        }

        validation = processor.validate_analysis(analysis_result)

        assert validation["valid"] == True
        assert validation["validations"]["intent_valid"] == True
        assert validation["validations"]["entities_found"] == True
        assert validation["validations"]["sql_ready"] == True
        assert validation["recommendation"] == "proceed_to_sql"

    def test_validate_analysis_low_confidence(self):
        """Test analysis validation - low confidence"""
        processor = NLPProcessor()

        analysis_result = {
            "intent": {"confidence": 0.5},  # Low confidence
            "entities": {
                "tables": [{"table": "customers"}],
                "metadata": {"complexity": "simple"}
            },
            "analysis_metadata": {"sql_ready": False}
        }

        validation = processor.validate_analysis(analysis_result)

        assert validation["valid"] == False
        assert validation["validations"]["intent_valid"] == False
        assert validation["recommendation"] == "clarify_intent"

    def test_get_processing_stats(self):
        """Test processing statistics"""
        processor = NLPProcessor()

        # Initial stats
        stats = processor.get_processing_stats()
        assert stats["total_processed"] == 0
        assert stats["successful_analyses"] == 0
        assert stats["success_rate"] == 0
        assert "models_loaded" in stats

        # Update stats manually for testing
        processor.processed_queries = 10
        processor.successful_analyses = 8

        stats = processor.get_processing_stats()
        assert stats["total_processed"] == 10
        assert stats["successful_analyses"] == 8
        assert stats["success_rate"] == 80.0

    def test_get_system_info(self):
        """Test system information"""
        processor = NLPProcessor()

        info = processor.get_system_info()

        assert "nlp_processor_version" in info
        assert "components" in info
        assert "supported_intents" in info
        assert "supported_entities" in info

        assert info["supported_entities"] == ["tables", "time_filters"]
        assert "intent_classifier" in info["components"]
        assert "entity_extractor" in info["components"]
        assert "berturk_wrapper" in info["components"]

    def test_get_recommendation_scenarios(self):
        """Test recommendation logic for different scenarios"""
        processor = NLPProcessor()

        # Valid intent and entities
        validations = {
            "intent_valid": True,
            "entities_found": True,
            "sql_ready": True,
            "complexity_acceptable": True
        }
        recommendation = processor._get_recommendation(validations)
        assert recommendation == "proceed_to_sql"

        # Invalid intent
        validations = {
            "intent_valid": False,
            "entities_found": True,
            "sql_ready": False,
            "complexity_acceptable": True
        }
        recommendation = processor._get_recommendation(validations)
        assert recommendation == "clarify_intent"

        # No entities found
        validations = {
            "intent_valid": True,
            "entities_found": False,
            "sql_ready": False,
            "complexity_acceptable": True
        }
        recommendation = processor._get_recommendation(validations)
        assert recommendation == "specify_tables"

        # Other issues
        validations = {
            "intent_valid": True,
            "entities_found": True,
            "sql_ready": False,
            "complexity_acceptable": False
        }
        recommendation = processor._get_recommendation(validations)
        assert recommendation == "review_query"