"""
Entity Extractor Tests
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.nlp.entity_extractor import EntityExtractor, create_entity_extractor


class TestEntityExtractor:
    """Test entity extractor functionality"""

    def test_extractor_initialization(self):
        """Test extractor initializes correctly"""
        extractor = EntityExtractor()

        assert extractor.similarity_threshold == 0.65
        assert "customers" in extractor.table_patterns
        assert "current_month" in extractor.time_patterns
        assert len(extractor.table_patterns) > 0
        assert len(extractor.time_patterns) > 0

    def test_create_entity_extractor(self):
        """Test convenience function"""
        extractor = create_entity_extractor()

        assert isinstance(extractor, EntityExtractor)

    def test_extract_empty_text(self):
        """Test extraction with empty text"""
        extractor = EntityExtractor()

        with pytest.raises(ValueError, match="Text input cannot be empty"):
            extractor.extract("")

        with pytest.raises(ValueError, match="Text input cannot be empty"):
            extractor.extract("   ")

    @patch('src.nlp.entity_extractor.BERTurkWrapper')
    def test_extract_basic_structure(self, mock_berturk, sample_texts):
        """Test extraction returns correct structure"""
        # Mock BERTurk wrapper
        mock_berturk_instance = MagicMock()
        mock_berturk.return_value = mock_berturk_instance
        mock_berturk_instance.get_embeddings.return_value = np.random.rand(768)

        extractor = EntityExtractor()

        for text in sample_texts:
            result = extractor.extract(text)

            # Check structure
            assert "text" in result
            assert "tables" in result
            assert "time_filters" in result
            assert "metadata" in result

            # Check types
            assert isinstance(result["tables"], list)
            assert isinstance(result["time_filters"], list)
            assert isinstance(result["metadata"], dict)

            # Check metadata fields
            assert "total_entities" in result["metadata"]
            assert "has_time_filter" in result["metadata"]
            assert "requires_join" in result["metadata"]
            assert "complexity" in result["metadata"]

    @patch('src.nlp.entity_extractor.BERTurkWrapper')
    def test_extract_customer_table_detection(self, mock_berturk):
        """Test customer table detection"""
        # Mock BERTurk wrapper to return high similarity for customer patterns
        mock_berturk_instance = MagicMock()
        mock_berturk.return_value = mock_berturk_instance

        def mock_get_embeddings(text):
            # Return high similarity embedding for customer-related patterns
            if any(word in text.lower() for word in ["müşteri", "customer"]):
                return np.ones(768)  # High similarity pattern
            return np.zeros(768)  # Low similarity pattern

        mock_berturk_instance.get_embeddings.side_effect = mock_get_embeddings

        extractor = EntityExtractor()
        result = extractor.extract("müşteri sayısını hesapla")

        # Should detect customer table
        assert len(result["tables"]) > 0
        customer_table = next((t for t in result["tables"] if t["table"] == "customers"), None)
        assert customer_table is not None
        assert "confidence" in customer_table
        assert customer_table["confidence"] > extractor.similarity_threshold

    @patch('src.nlp.entity_extractor.BERTurkWrapper')
    def test_extract_time_filter_detection(self, mock_berturk):
        """Test time filter detection"""
        # Mock BERTurk wrapper
        mock_berturk_instance = MagicMock()
        mock_berturk.return_value = mock_berturk_instance

        def mock_get_embeddings(text):
            # Return high similarity for time patterns
            if any(phrase in text.lower() for phrase in ["bu ay", "current month"]):
                return np.ones(768)  # High similarity pattern
            return np.zeros(768)  # Low similarity pattern

        mock_berturk_instance.get_embeddings.side_effect = mock_get_embeddings

        extractor = EntityExtractor()
        result = extractor.extract("bu ayın sipariş listesi")

        # Should detect time filter
        assert len(result["time_filters"]) > 0
        current_month_filter = next((t for t in result["time_filters"] if t["period"] == "current_month"), None)
        assert current_month_filter is not None
        assert "sql_condition" in current_month_filter
        assert "confidence" in current_month_filter

    @patch('src.nlp.entity_extractor.BERTurkWrapper')
    def test_extract_multiple_entities(self, mock_berturk):
        """Test extraction with multiple entities"""
        # Mock BERTurk wrapper
        mock_berturk_instance = MagicMock()
        mock_berturk.return_value = mock_berturk_instance

        def mock_get_embeddings(text):
            # Return high similarity for both table and time patterns
            if any(word in text.lower() for word in ["sipariş", "order"]):
                return np.ones(768)
            elif any(phrase in text.lower() for phrase in ["bu ay", "current"]):
                return np.ones(768)
            return np.zeros(768)

        mock_berturk_instance.get_embeddings.side_effect = mock_get_embeddings

        extractor = EntityExtractor()
        result = extractor.extract("bu ayın sipariş listesi")

        # Should detect both table and time filter
        assert len(result["tables"]) > 0
        assert len(result["time_filters"]) > 0
        assert result["metadata"]["has_time_filter"] == True
        assert result["metadata"]["total_entities"] >= 2

    @patch('src.nlp.entity_extractor.BERTurkWrapper')
    def test_extract_no_entities(self, mock_berturk):
        """Test extraction with no matching entities"""
        # Mock BERTurk wrapper to return low similarity
        mock_berturk_instance = MagicMock()
        mock_berturk.return_value = mock_berturk_instance
        mock_berturk_instance.get_embeddings.return_value = np.zeros(768)  # Low similarity

        extractor = EntityExtractor()
        result = extractor.extract("random meaningless text")

        # Should detect no entities
        assert len(result["tables"]) == 0
        assert len(result["time_filters"]) == 0
        assert result["metadata"]["total_entities"] == 0
        assert result["metadata"]["complexity"] == "no_entities"

    def test_get_sql_condition(self):
        """Test SQL condition generation"""
        extractor = EntityExtractor()

        # Test different time periods
        test_cases = {
            "current_month": "DATE_TRUNC('month', CURRENT_DATE)",
            "current_year": "DATE_TRUNC('year', CURRENT_DATE)",
            "last_month": "DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')",
            "last_year": "DATE_TRUNC('year', CURRENT_DATE - INTERVAL '1 year')",
            "today": "CURRENT_DATE",
            "last_week": "DATE_TRUNC('week', CURRENT_DATE - INTERVAL '1 week')",
            "current_week": "DATE_TRUNC('week', CURRENT_DATE)"
        }

        for period, expected_sql in test_cases.items():
            result = extractor._get_sql_condition(period)
            assert result == expected_sql

        # Test unknown period
        result = extractor._get_sql_condition("unknown_period")
        assert result == "CURRENT_DATE"

    def test_assess_complexity(self):
        """Test complexity assessment"""
        extractor = EntityExtractor()

        # Test different complexity levels
        assert extractor._assess_complexity([], []) == "no_entities"

        # Simple: 1 entity
        assert extractor._assess_complexity([{"table": "customers"}], []) == "simple"
        assert extractor._assess_complexity([], [{"period": "current_month"}]) == "simple"

        # Medium: 2-3 entities
        assert extractor._assess_complexity(
            [{"table": "customers"}],
            [{"period": "current_month"}]
        ) == "medium"

        # Complex: 4+ entities
        tables = [{"table": "customers"}, {"table": "orders"}]
        time_filters = [{"period": "current_month"}, {"period": "last_month"}]
        assert extractor._assess_complexity(tables, time_filters) == "complex"

    def test_calculate_similarity(self):
        """Test similarity calculation"""
        extractor = EntityExtractor()

        # Test identical vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = extractor._calculate_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001  # Should be close to 1.0

        # Test orthogonal vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = extractor._calculate_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001  # Should be close to 0.0

        # Test zero vectors
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        similarity = extractor._calculate_similarity(vec1, vec2)
        assert similarity == 0.0  # Should handle zero norm

    @patch('src.nlp.entity_extractor.BERTurkWrapper')
    def test_get_extraction_summary(self, mock_berturk):
        """Test extraction summary"""
        # Mock BERTurk wrapper
        mock_berturk_instance = MagicMock()
        mock_berturk.return_value = mock_berturk_instance
        mock_berturk_instance.get_embeddings.return_value = np.random.rand(768)

        extractor = EntityExtractor()

        # Mock extract method
        with patch.object(extractor, 'extract') as mock_extract:
            mock_extract.return_value = {
                "tables": [{"table": "customers", "confidence": 0.9}],
                "time_filters": [{"period": "current_month", "confidence": 0.8}],
                "metadata": {
                    "complexity": "medium",
                    "requires_join": False
                }
            }

            summary = extractor.get_extraction_summary("müşteri sayısını hesapla")

            assert "input_text" in summary
            assert "tables_detected" in summary
            assert "time_filters_detected" in summary
            assert "complexity" in summary
            assert "top_table" in summary
            assert "requires_join" in summary

            assert summary["tables_detected"] == 1
            assert summary["time_filters_detected"] == 1
            assert summary["complexity"] == "medium"

    def test_table_patterns_coverage(self):
        """Test that all expected table patterns are defined"""
        extractor = EntityExtractor()

        expected_tables = ["customers", "products", "orders", "categories", "suppliers"]

        for table in expected_tables:
            assert table in extractor.table_patterns
            assert len(extractor.table_patterns[table]) > 0

    def test_time_patterns_coverage(self):
        """Test that all expected time patterns are defined"""
        extractor = EntityExtractor()

        expected_time_periods = [
            "current_month", "current_year", "last_month",
            "last_year", "today", "last_week", "current_week"
        ]

        for period in expected_time_periods:
            assert period in extractor.time_patterns
            assert len(extractor.time_patterns[period]) > 0