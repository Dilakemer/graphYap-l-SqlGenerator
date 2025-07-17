"""
Test configuration and fixtures
"""
import pytest
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_texts():
    """Sample Turkish texts for testing"""
    return [
        "müşteri sayısını hesapla",
        "bu ayın sipariş listesi",
        "geçen yılın ürün sayısı",
        "kategori bilgilerini göster",
        "satış toplamını bul",
        "tedarikçi listesini göster"
    ]

@pytest.fixture
def expected_intents():
    """Expected intent classifications"""
    return {
        "müşteri sayısını hesapla": "COUNT",
        "bu ayın sipariş listesi": "SELECT",
        "geçen yılın ürün sayısı": "COUNT",
        "kategori bilgilerini göster": "SELECT",
        "satış toplamını bul": "SUM"
    }

@pytest.fixture
def expected_tables():
    """Expected table extractions"""
    return {
        "müşteri sayısını hesapla": "customers",
        "bu ayın sipariş listesi": "orders",
        "geçen yılın ürün sayısı": "products",
        "kategori bilgilerini göster": "categories",
        "satış toplamını bul": "orders"
    }

@pytest.fixture
def mock_embedding():
    """Fixed mock embedding for deterministic testing"""
    np.random.seed(42)  # Fixed seed for reproducibility
    return np.random.rand(768).astype(np.float32)

@pytest.fixture
def mock_batch_embeddings():
    """Fixed mock batch embeddings for deterministic testing"""
    np.random.seed(42)  # Fixed seed for reproducibility
    return np.random.rand(5, 768).astype(np.float32)

@pytest.fixture
def high_similarity_embedding():
    """High similarity embedding for positive test cases"""
    return np.ones(768, dtype=np.float32)

@pytest.fixture
def low_similarity_embedding():
    """Low similarity embedding for negative test cases"""
    return np.zeros(768, dtype=np.float32)

@pytest.fixture
def sample_intent_result():
    """Sample intent classification result"""
    return {
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

@pytest.fixture
def sample_entity_result():
    """Sample entity extraction result"""
    return {
        "text": "müşteri sayısını hesapla",
        "tables": [
            {
                "table": "customers",
                "confidence": 0.9,
                "matched_pattern": "müşteri sayısı"
            }
        ],
        "time_filters": [],
        "metadata": {
            "total_entities": 1,
            "has_time_filter": False,
            "requires_join": False,
            "complexity": "simple"
        }
    }

@pytest.fixture
def sample