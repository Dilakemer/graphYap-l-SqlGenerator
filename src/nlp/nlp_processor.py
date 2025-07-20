# src/nlp/nlp_processor.py
"""
NLP Processor - Refactored for NER-only approach
Coordinates NER-based entity and intent extraction
No longer uses separate Intent Classifier
"""

from entity_extractor import EntityExtractor
from berturk_wrapper import BERTurkWrapper


class NLPProcessor:
    """
    Main NLP Processor - Refactored Version
    Uses only NER model for both entity and intent detection
    Simplified architecture with better performance
    """

    def __init__(self):
        """Initialize with NER-based entity extractor only"""
        self.entity_extractor = EntityExtractor()
        self.berturk = BERTurkWrapper()  # Keep for potential future use

        # Processing statistics
        self.processed_queries = 0
        self.successful_analyzes = 0

        print("🤖 NLP Processor initialized (NER-only mode)")
        print(f"📊 Entity Extractor ready: {self.entity_extractor.is_ready()}")

    def analyze(self, text):
        """
        Analyze Turkish text for SQL Generation using NER model

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

            # Extract entities and intents using NER model
            extraction_result = self.entity_extractor.extract(text)

            # Format result to match expected interface
            analysis_result = {
                "text": text,
                "intent": self._format_intent_output(extraction_result),
                "entities": self._format_entities_output(extraction_result),
                "analysis_metadata": self._format_metadata_output(extraction_result)
            }

            # Track successful analysis
            if extraction_result["metadata"]["processing_status"] == "success":
                self.successful_analyzes += 1
                analysis_result["analysis_metadata"]["processing_status"] = "success"
            else:
                analysis_result["analysis_metadata"]["processing_status"] = "error"

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
                    "extraction_method": "ner_model"
                }
            }

    def _format_intent_output(self, extraction_result):
        """Format intent output to match expected interface"""
        primary_intent = extraction_result.get("primary_intent")

        if primary_intent:
            # Map INTENT_X to standard format
            intent_mapping = {
                "INTENT_SELECT": "SELECT",
                "INTENT_COUNT": "COUNT",
                "INTENT_SUM": "SUM",
                "INTENT_AVG": "AVG"
            }

            # Clean the intent type - remove INTENT_ prefix if present
            raw_intent = primary_intent["type"]
            if raw_intent.startswith("INTENT_"):
                clean_intent = intent_mapping.get(raw_intent, raw_intent.replace("INTENT_", ""))
            else:
                clean_intent = raw_intent

            # Create all_confidences from detected intents
            all_confidences = {}
            for intent_entity in extraction_result.get("intents", []):
                intent_type = intent_entity["intent"]
                if intent_type.startswith("INTENT_"):
                    mapped_intent = intent_mapping.get(intent_type, intent_type.replace("INTENT_", ""))
                else:
                    mapped_intent = intent_type
                all_confidences[mapped_intent] = intent_entity["confidence"]

            return {
                "type": clean_intent,
                "confidence": primary_intent["confidence"],
                "all_confidences": all_confidences,
                "extraction_method": "ner_model",
                "matched_pattern": primary_intent.get("matched_pattern", "")
            }
        else:
            return {
                "type": "UNKNOWN",
                "confidence": 0.0,
                "all_confidences": {},
                "extraction_method": "ner_model",
                "matched_pattern": ""
            }

    def _format_entities_output(self, extraction_result):
        """Format entities output to match expected interface"""
        return {
            "tables": extraction_result.get("tables", []),
            "time_filters": extraction_result.get("time_filters", []),
            "numbers": extraction_result.get("numbers", []),
            "metadata": extraction_result.get("metadata", {})
        }

    def _format_metadata_output(self, extraction_result):
        """Format metadata output to match expected interface"""
        metadata = extraction_result.get("metadata", {})

        return {
            "processing_status": metadata.get("processing_status", "unknown"),
            "extraction_method": "ner_model",
            "intent_confidence": extraction_result.get("primary_intent", {}).get("confidence", 0.0),
            "entity_complexity": metadata.get("complexity", "unknown"),
            "confidence_level": metadata.get("confidence_level", "unknown"),
            "requires_join": metadata.get("requires_join", False),
            "has_time_filter": metadata.get("has_time_filter", False),
            "has_intent": metadata.get("has_intent", False),
            "sql_ready": metadata.get("sql_ready", False),
            "total_entities": metadata.get("total_entities", 0),
            "table_count": metadata.get("table_count", 0),
            "time_filter_count": metadata.get("time_filter_count", 0),
            "intent_count": metadata.get("intent_count", 0)
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
                results.append({
                    "text": text,
                    "intent": {"type": "ERROR", "confidence": 0.0},
                    "entities": {"tables": [], "time_filters": []},
                    "analysis_metadata": {
                        "processing_status": "error",
                        "error_message": str(e),
                        "sql_ready": False,
                        "extraction_method": "ner_model"
                    }
                })

        return results

    def get_query_context(self, analysis_result):
        """
        Extract query context for SQL generation

        Args:
            analysis_result: Result from analyze method
        Returns:
            Dictionary with query context
        """
        if not analysis_result["analysis_metadata"]["sql_ready"]:
            return {
                "ready": False,
                "reason": "Analysis not ready for SQL generation",
                "extraction_method": "ner_model"
            }

        # Extract primary table
        tables = analysis_result["entities"]["tables"]
        primary_table = tables[0]["table"] if tables else None

        # Extract time filter if exists
        time_filters = analysis_result["entities"]["time_filters"]
        time_filter = time_filters[0] if time_filters else None

        return {
            "ready": True,
            "intent": analysis_result["intent"]["type"],
            "primary_table": primary_table,
            "time_filter": time_filter,
            "requires_join": analysis_result["analysis_metadata"]["requires_join"],
            "confidence": analysis_result["intent"]["confidence"],
            "extraction_method": "ner_model",
            "complexity": analysis_result["analysis_metadata"]["entity_complexity"]
        }

    def validate_analysis(self, analysis_result):
        """
        Validate analysis result

        Args:
            analysis_result: Result from analyze() method
        Returns:
            Dictionary with validation results
        """
        metadata = analysis_result["analysis_metadata"]

        validations = {
            "intent_valid": analysis_result["intent"]["confidence"] > 0.7,
            "entities_found": len(analysis_result["entities"]["tables"]) > 0,
            "sql_ready": metadata["sql_ready"],
            "complexity_acceptable": metadata["entity_complexity"] in ["simple", "medium"],
            "confidence_acceptable": metadata["confidence_level"] in ["high", "very_high", "medium"],
            "extraction_method": metadata["extraction_method"] == "ner_model"
        }

        return {
            "valid": all(validations.values()),
            "validations": validations,
            "recommendation": self._get_recommendation(validations),
            "extraction_method": "ner_model"
        }

    def _get_recommendation(self, validations):
        """Get recommendation based on validation results"""
        if validations["intent_valid"] and validations["entities_found"] and validations["confidence_acceptable"]:
            return "proceed_to_sql"
        elif not validations["intent_valid"]:
            return "clarify_intent"
        elif not validations["entities_found"]:
            return "specify_tables"
        elif not validations["confidence_acceptable"]:
            return "rephrase_query"
        else:
            return "review_query"

    def get_processing_stats(self):
        """Get processing statistics"""
        success_rate = (
            (self.successful_analyzes / self.processed_queries * 100)
            if self.processed_queries > 0
            else 0
        )

        # Get entity extractor stats
        extractor_stats = self.entity_extractor.get_statistics()

        return {
            "total_processed": self.processed_queries,
            "successful_analyzes": self.successful_analyzes,
            "success_rate": round(success_rate, 2),
            "extraction_method": "ner_model",
            "models_loaded": {
                "entity_extractor": self.entity_extractor.is_ready(),
                "ner_model": extractor_stats.get("model_loaded", False),
                "berturk": self.berturk.is_loaded()
            },
            "extractor_stats": extractor_stats
        }

    def get_system_info(self):
        """Get system information"""
        extractor_stats = self.entity_extractor.get_statistics()

        return {
            "nlp_processor_version": "2.0.0",
            "architecture": "ner_only",
            "extraction_method": "trained_ner_model",
            "components": {
                "entity_extractor": "ner_based",
                "intent_classifier": "integrated_in_ner",  # No longer separate
                "berturk_wrapper": self.berturk.get_model_info()
            },
            "supported_intents": ["SELECT", "COUNT", "SUM", "AVG"],
            "supported_entities": ["tables", "time_filters", "numbers"],
            "model_info": extractor_stats.get("model_info", {})
        }

    def test_extraction(self, test_texts=None):
        """Test extraction on sample texts"""
        if test_texts is None:
            test_texts = [
                "Bu ayın müşteri sayısı",
                "Geçen yıl sipariş toplamı",
                "Son 30 gün ürün listesi",
                "2022 3. çeyrek çalışan maaşları"
            ]

        print("🧪 Testing NLP Processor extraction...")

        for text in test_texts:
            print(f"\n🔍 Testing: '{text}'")

            try:
                result = self.analyze(text)

                print(f"  Intent: {result['intent']['type']} (confidence: {result['intent']['confidence']:.3f})")
                print(f"  Tables: {len(result['entities']['tables'])}")
                print(f"  Time filters: {len(result['entities']['time_filters'])}")
                print(f"  SQL ready: {result['analysis_metadata']['sql_ready']}")
                print(f"  Complexity: {result['analysis_metadata'].get('entity_complexity', 'unknown')}")

            except Exception as e:
                print(f"  ❌ Error: {e}")


def create_nlp_processor():
    """Factory function to create NLP processor"""
    return NLPProcessor()


# Test function
if __name__ == "__main__":
    print("🧪 Testing Refactored NLP Processor...")

    processor = NLPProcessor()
    processor.test_extraction()

    # Show system info
    print(f"\n📊 System Info:")
    info = processor.get_system_info()
    print(f"  Version: {info['nlp_processor_version']}")
    print(f"  Architecture: {info['architecture']}")
    print(f"  Extraction method: {info['extraction_method']}")

    # Show statistics
    stats = processor.get_processing_stats()
    print(f"\n📈 Statistics:")
    print(f"  Success rate: {stats['success_rate']}%")
    print(f"  Models loaded: {stats['models_loaded']}")