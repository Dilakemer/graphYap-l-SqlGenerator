"""
Entity Extractor - Simplified version.
Detects tables and basic time filters only
"""

from jinja2 import pass_context

from .berturk_wrapper import BERTurkWrapper
import numpy as np


class EntityExtractor:

    def __init__(self):
        self.berturk = BERTurkWrapper()

        # Load basic patterns
        self.table_patterns = self._load_table_patterns()
        self.time_patterns = self._load_time_patterns()

        # Similarity Threshold = 0.65
        self.similarity_threshold = 0.65

    def _load_table_patterns(self):
        """Load table detection patterns"""
        return {
            "customers": [
                "müşteri sayısı",
                "müşteri listesi",
                "müşteri bilgileri",
                "müşteri kayıtları",
                "firma sayısı",
                "şirket listesi",
                "client bilgileri",
                "müşterilerin listesi",
            ],
            "products": [
                "ürün sayısı",
                "ürün listesi",
                "ürün bilgileri",
                "mal sayısı",
                "product listesi",
                "stok bilgileri",
                "ürün kayıtları",
            ],
            "orders": [
                "sipariş sayısı",
                "sipariş listesi",
                "order bilgileri",
                "satış sayısı",
                "satış listesi",
                "alışveriş kayıtları",
                "sipariş kayıtları",
            ],
            "categories": [
                "kategori sayısı",
                "kategori listesi",
                "grup bilgileri",
                "tür sayısı",
                "sınıf bilgileri",
                "kategori kayıtları",
            ],
            "suppliers": [
                "tedarikçi sayısı",
                "tedarikçi listesi",
                "supplier bilgileri",
                "sağlayıcı sayısı",
                "tedarikçi kayıtları",
            ],
        }

    def _load_time_patterns(self):
        """Load time filter patterns"""
        return {
            "current_month": [
                "bu ayın",
                "bu ayki",
                "mevcut ayın",
                "şu anki ayın",
                "bu ay",
            ],
            "current_year": [
                "bu yılın",
                "bu yıla ait",
                "mevcut yılın",
                "şu anki yılın",
                "bu yıl",
            ],
            "last_month": ["geçen ayın", "önceki ayın", "geçtiğimiz ayın", "geçen ay"],
            "last_year": [
                "geçen yılın",
                "önceki yılın",
                "geçtiğimiz yılın",
                "geçen yıl",
            ],
            "today": ["bugünkü", "bugün", "bu günkü"],
            "last_week": [
                "geçen haftanın",
                "önceki haftanın",
                "geçtiğimiz haftanın",
                "geçen hafta",
            ],
            "current_week": ["bu haftanın", "bu hafta", "mevcut haftanın"],
        }

    def extract(self, text):
        """
        Extract entities from text.

        Args: Input text
        Returns: Dictionary with extracted entities
        """

        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        # Generate text embedding
        text_embedding = self.berturk.get_embeddings(text)

        # Extract different entity types
        tables = self._extract_tables(text_embedding)
        time_filters = self._extract_time_filters(text_embedding)

        return {
            "text": text,
            "tables": tables,
            "time_filters": time_filters,
            "metadata": {
                "total_entities": len(tables) + len(time_filters),
                "has_time_filter": len(time_filters) > 0,
                "requires_join": len(tables) > 1,
                "complexity": self._assess_complexity(tables, time_filters),
            },
        }

    def _extract_tables(self, text_embedding):
        """
        Extract table entities using semantic similarity
        """

        detected_tables = []

        for table_name, patterns in self.table_patterns.items():
            max_similarity = 0
            best_pattern = None

            for pattern in patterns:
                pattern_embedding = self.berturk.get_embeddings(pattern)
                similarity = self._calculate_similarity(
                    text_embedding, pattern_embedding
                )

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pattern = pattern

            if max_similarity > self.similarity_threshold:
                table_entity = {
                    "table": table_name,
                    "confidence": round(max_similarity, 3),
                    "matched_pattern": best_pattern,
                }
                detected_tables.append(table_entity)

        detected_tables.sort(key=lambda x: x["confidence"], reverse=True)

    def _extract_time_filters(self, text_embedding):
        """
        Extract time filters using semantic similarity
        """
        detected_time_filters = []

        for time_period, patterns in self.time_patterns.items():
            max_similarity = 0
            best_pattern = None

            for pattern in patterns:
                pattern_embedding = self.berturk.get_embeddings(pattern)
                similarity = self._calculate_similarity(
                    text_embedding, pattern_embedding
                )

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pattern = pattern

            if max_similarity > self.similarity_threshold:
                time_filter = {
                    "period": time_period,
                    "confidence": round(max_similarity, 3),
                    "matched_pattern": best_pattern,
                    "sql_condition": self._get_sql_condition(time_period),
                }
                detected_time_filters.append(time_filter)

        detected_time_filters.sort(key=lambda x: x["confidence"], reverse=True)
        return detected_time_filters

    def _calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between embeddings
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _get_sql_condition(self, time_period):
        """
        Get SQL condition for time period. This method generates the necessary time interval
        in line with the _extract_time_filters()"""

        conditions = {
            "current_month": "DATE_TRUNC('month', CURRENT_DATE)",
            "current_year": "DATE_TRUNC('month', CURRENT_DATE)",
            "last_month": "DATE_TRUNC('month', CURRENT_DATE)",
            "last_year": "DATE_TRUNC('month', CURRENT_DATE)",
            "today": "DATE_TRUNC('month', CURRENT_DATE)",
            "last_week": "DATE_TRUNC('week', CURRENT_DATE - INTERVAL)",
            "current_week": "DATE_TRUNC('week', CURRENT_DATE)",
        }

    def _assess_complexity(self, tables, time_filters):
        """
        Assess query complexity based on entities
        """
        total_entities = len(tables) + len(time_filters)

        if total_entities == 0:
            return "no entities"
        elif total_entities == 1:
            return "simple"
        elif total_entities <= 3:
            return "medium"
        else:
            return "complex"

    def get_extraction_summary(self, text):
        extraction = self.extract(text)

        return {
            "input_text": text,
            "tables_detected": len(extraction["time_filters"]),
            "time_filters_detected": len(extraction["time_filters"]),
            "complexity": extraction["metadata"]["complexity"],
            "top_table": (
                extraction["tables"][0]["table"] if extraction["tables"] else None
            ),
            "top_time_filter": (
                extraction["time_filters"][0]["period"]
                if extraction["time_filters"]
                else None
            ),
            "requires_join": extraction["metadata"]["requires_join"],
        }


def create_entity_extractor():
    """Create EntityExtractor instance"""
    return EntityExtractor()
