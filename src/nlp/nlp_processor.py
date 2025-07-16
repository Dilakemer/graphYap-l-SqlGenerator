"""
Coordinates Intent Classification and Entity Extractor
"""

from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor
from .berturk_wrapper import BERTurkWrapper


class NLPProcessor:
    """
    Main NLP Processor that coordinates all NLP components.
    Integrates intent classification and entity extractor
    """

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.berturk = BERTurkWrapper()

        # Processing statistics
        self.processed_queries = 0
        self.successful_analyzes = 0

    def analyze(self, text):
        """
        Analyze Turkish text for SQL Generation

        Args:
            text: Turkish text input
        Returns:
            Dictionary with complete NLP analysis
        """

        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        try:
            # Track processing
            self.processed_queries += 1

            # Step 1: Intent Classification
            intent_result = self.intent_classifier.predict(text)

            # Step 2: Entity Extraction
            entity_result = self.entity_extractor.extract(text)

            # Step 3: Combine results
            analysis_result = {
                "text": text,
                "intent": {
                    "type": intent_result["intent"],
                    "confidence": intent_result["confidence"],
                    "all_confidences": intent_result["all_confidences"],
                },
                "entities": {
                    "tables": entity_result["tables"],
                    "time_filters": entity_result["time_filters"],
                    "metadata": entity_result["metadata"],
                },
                "analysis_metadata": {
                    "processing_status": "success",
                    "intent_confidence": intent_result["confidence"],
                    "entity_complexity": entity_result["metadata"]["complexity"],
                    "requires_join": entity_result["metadata"]["requires_join"],
                    "has_time_filter": entity_result["metadata"]["has_time_filter"],
                    "sql_ready": self._is_sql_ready(intent_result, entity_result),
                },
            }

            # Track successful analysis
            self.successful_analyzes += 1

            return analysis_result

        except Exception as e:

            # Handle errors gracefully
            return {
                "text": text,
                "intent": {"type": "UNKNOWN", "confidence": 0.0},
                "entities": {"tables": [], "time_filters": [], "metadata": {}},
                "analysis_metadata": {
                    "processing_status": "error",
                    "error_message": str(e),
                    "sql_ready": False,
                },
            }

    def analyze_batch(self, texts):
        """
        Analyze multiple texts

        Args:
            texts: List of Turkish text inputs

        Returns:
            List of analysis results
        """
        if not texts:
            raise ValueError("Text list cannot be empty")

        results = []
        for text in texts:
            try:
                result = self.analyze(text)
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "text": text,
                        "intent": {"type": "ERROR", "confidence": 0.0},
                        "entities": {"tables": [], "time_filters": []},
                        "analysis_metadata": {
                            "processing_status": "error",
                            "error_message": str(e),
                            "sql_ready": False,
                        },
                    }
                )

        return results

    def _is_sql_ready(self, intent_result, entity_result):
        """
        Check if analysis is ready for SQL generation

        Args:
            intent_result: Intent Classification result
            entity_result: Entity Extraction result
        Returns:
            Boolean indicating SQL readiness
        """

        if intent_result["confidence"] < 0.6:
            return False

        # Check if we have tables (required for SQL)
        if not entity_result["tables"]:
            return False

        # Check if intent is supported
        supported_intents = ["SELECT", "COUNT", "SUM", "AVG"]

        if intent_result["intent"] not in supported_intents:
            return False

        return True

    def get_query_context(self, analysis_result):
        """
        Extract query context for SQL generation.

        Args:
            analysis_result: Result from analyze method
        Returns:
            Dictionary with query context
        """

        if not analysis_result["analysis_metadata"]["sql_ready"]:
            return {"ready": False, "reason": "Analysis not ready for SQL generation"}

        # Extract primary table
        primary_table = analysis_result["entities"]["tables"][0]["table"]

        # Extract time filter if exists
        time_filter = None
        if analysis_result["entities"]["time_filters"]:
            time_filter = analysis_result["entities"]["time_filters"][0]

        return {
            "ready": True,
            "intent": analysis_result["intent"]["type"],
            "primary_table": primary_table,
            "time_filter": time_filter,
            "requires_join": analysis_result["analysis_metadata"]["requires_join"],
            "confidence": analysis_result["intent"]["confidence"],
        }

    def validate_analysis(self, analysis_result):
        """
        Validate analysis result

        Args:
            analysis_result: Result from analyze() method

        Returns:
            Dictionary with validation results
        """
        validations = {
            "intent_valid": analysis_result["intent"]["confidence"] > 0.6,
            "entities_found": len(analysis_result["entities"]["tables"]) > 0,
            "sql_ready": analysis_result["analysis_metadata"]["sql_ready"],
            "complexity_acceptable": analysis_result["entities"]["metadata"][
                "complexity"
            ]
            in ["simple", "medium"],
        }

        return {
            "valid": all(validations.values()),
            "validations": validations,
            "recommendation": self._get_recommendation(validations),
        }

    def _get_recommendation(self, validations):
        """Get recommendation based on validation results"""
        if validations["intent_valid"] and validations["entities_found"]:
            return "proceed_to_sql"
        elif not validations["intent_valid"]:
            return "clarify_intent"
        elif not validations["entities_found"]:
            return "specify_tables"
        else:
            return "review_query"

    def get_processing_stats(self):
        """Get processing statistics"""
        success_rate = (
            (self.successful_analyses / self.processed_queries * 100)
            if self.processed_queries > 0
            else 0
        )

        return {
            "total_processed": self.processed_queries,
            "successful_analyses": self.successful_analyses,
            "success_rate": round(success_rate, 2),
            "models_loaded": {
                "intent_classifier": self.intent_classifier.is_trained,
                "entity_extractor": True,
                "berturk": self.berturk.is_loaded(),
            },
        }

    def get_system_info(self):
        """Get system information"""
        return {
            "nlp_processor_version": "1.0.0",
            "components": {
                "intent_classifier": self.intent_classifier.get_model_info(),
                "entity_extractor": "semantic_similarity_based",
                "berturk_wrapper": self.berturk.get_model_info(),
            },
            "supported_intents": self.intent_classifier.get_supported_intents(),
            "supported_entities": ["tables", "time_filters"],
        }


def create_nlp_processor():
    """Create NLP processor instance"""
    return NLPProcessor()
