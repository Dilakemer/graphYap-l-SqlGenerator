# src/nlp/entity_extractor.py
"""
NER-based Entity Extractor for Turkish NLP-SQL Project
Uses trained Turkish NER model for high-accuracy entity detection
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our trained NER model
from ner_model.turkish_ner import TurkishNER


class EntityExtractor:
    """
    NER-based Entity Extractor
    Uses trained Turkish NER model for entity and intent detection
    Replaces similarity-based approach with deep learning
    """
    
    def __init__(self):
        """Initialize with trained NER model"""
        self.ner_model = TurkishNER()
        self.is_loaded = False
        
        # Statistics
        self.extracted_queries = 0
        self.successful_extractions = 0
        
        # Load trained model
        self._load_ner_model()

    def _load_ner_model(self):
        """Load the trained NER model"""
        print("🔄 Loading trained NER model...")
        
        if self.ner_model.load_model():
            self.is_loaded = True
            print("✅ NER model loaded successfully!")
        else:
            print("⚠️ Trained model not found, initializing fresh model...")
            if self.ner_model.initialize_model():
                self.is_loaded = True
                print("✅ Fresh NER model initialized!")
            else:
                print("❌ Failed to initialize NER model!")
                self.is_loaded = False

    def extract(self, text):
        """
        Extract entities and intents from text using trained NER model
        
        Args:
            text: Input Turkish text
            
        Returns:
            Dictionary with extracted entities, intents, and metadata
        """
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")
        
        if not self.is_loaded:
            raise RuntimeError("NER model not loaded. Cannot perform extraction.")
        
        self.extracted_queries += 1
        
        try:
            # Get all entities from NER model
            all_entities = self.ner_model.predict(text, return_confidence=True)
            
            # Separate entities by type
            tables = []
            time_filters = []
            intents = []
            numbers = []
            other_entities = []
            
            for entity in all_entities:
                label = entity["label"]
                
                if label.startswith("TABLE_"):
                    tables.append(self._format_table_entity(entity))
                elif label.startswith("TIME_"):
                    time_filters.append(self._format_time_entity(entity))
                elif label.startswith("INTENT_"):
                    intents.append(self._format_intent_entity(entity))
                elif label in ["TIME_NUMBER", "TIME_UNIT"]:
                    numbers.append(self._format_number_entity(entity))
                else:
                    other_entities.append(entity)
            
            # Sort by confidence
            tables.sort(key=lambda x: x["confidence"], reverse=True)
            time_filters.sort(key=lambda x: x["confidence"], reverse=True)
            intents.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Determine primary intent
            primary_intent = self._determine_primary_intent(intents)
            
            # Create metadata
            metadata = self._create_metadata(tables, time_filters, intents, all_entities)
            
            result = {
                "text": text,
                "tables": tables,
                "time_filters": time_filters,
                "intents": intents,
                "primary_intent": primary_intent,
                "numbers": numbers,
                "other_entities": other_entities,
                "all_entities": all_entities,
                "metadata": metadata
            }
            
            self.successful_extractions += 1
            return result
            
        except Exception as e:
            print(f"❌ Entity extraction failed: {e}")
            return {
                "text": text,
                "tables": [],
                "time_filters": [],
                "intents": [],
                "primary_intent": None,
                "numbers": [],
                "other_entities": [],
                "all_entities": [],
                "metadata": {
                    "processing_status": "error",
                    "error_message": str(e),
                    "total_entities": 0,
                    "complexity": "error"
                }
            }

    def _format_table_entity(self, entity):
        """Format table entity for consistency with old interface"""
        # Map NER label to table name
        table_name = self.ner_model._map_table_label_to_name(f"TABLE_{entity['label']}")
        
        return {
            "table": table_name,
            "confidence": entity["confidence"],
            "matched_pattern": entity["text"],
            "start": entity["start"],
            "end": entity["end"],
            "original_label": entity["label"]
        }

    def _format_time_entity(self, entity):
        """Format time entity for consistency with old interface"""
        # Map NER label to time period
        time_period = self.ner_model._map_time_label_to_period(f"TIME_{entity['label']}", entity["text"])
        
        return {
            "period": time_period,
            "confidence": entity["confidence"],
            "matched_pattern": entity["text"],
            "start": entity["start"],
            "end": entity["end"],
            "original_label": entity["label"]
        }

    def _format_intent_entity(self, entity):
        """Format intent entity"""
        # Map INTENT_X to intent type
        intent_mapping = {
            "INTENT_SELECT": "SELECT",
            "INTENT_COUNT": "COUNT", 
            "INTENT_SUM": "SUM",
            "INTENT_AVG": "AVG"
        }
        
        intent_type = intent_mapping.get(f"INTENT_{entity['label']}", entity['label'])
        
        return {
            "intent": intent_type,
            "confidence": entity["confidence"],
            "matched_pattern": entity["text"],
            "start": entity["start"],
            "end": entity["end"],
            "original_label": entity["label"]
        }

    def _format_number_entity(self, entity):
        """Format number/unit entities"""
        return {
            "type": entity["label"],  # TIME_NUMBER or TIME_UNIT
            "value": entity["text"],
            "confidence": entity["confidence"],
            "start": entity["start"],
            "end": entity["end"]
        }

    def _determine_primary_intent(self, intents):
        """Determine the primary intent from detected intents"""
        if not intents:
            return None
        
        # Return highest confidence intent
        primary = max(intents, key=lambda x: x["confidence"])
        
        return {
            "type": primary["intent"],
            "confidence": primary["confidence"],
            "matched_pattern": primary["matched_pattern"]
        }

    def _create_metadata(self, tables, time_filters, intents, all_entities):
        """Create metadata about the extraction"""
        return {
            "processing_status": "success",
            "extraction_method": "trained_ner_model",
            "total_entities": len(all_entities),
            "table_count": len(tables),
            "time_filter_count": len(time_filters),
            "intent_count": len(intents),
            "has_time_filter": len(time_filters) > 0,
            "has_intent": len(intents) > 0,
            "requires_join": len(tables) > 1,
            "complexity": self._assess_complexity(tables, time_filters, intents),
            "confidence_level": self._assess_confidence_level(all_entities),
            "sql_ready": self._is_sql_ready(tables, intents)
        }

    def _assess_complexity(self, tables, time_filters, intents):
        """Assess query complexity"""
        total_entities = len(tables) + len(time_filters) + len(intents)
        
        if total_entities == 0:
            return "no_entities"
        elif total_entities <= 2:
            return "simple"
        elif total_entities <= 4:
            return "medium"
        else:
            return "complex"

    def _assess_confidence_level(self, entities):
        """Assess overall confidence level"""
        if not entities:
            return "no_entities"
        
        confidences = [e["confidence"] for e in entities if e["confidence"] is not None]
        
        if not confidences:
            return "unknown"
        
        avg_confidence = sum(confidences) / len(confidences)
        
        if avg_confidence >= 0.9:
            return "very_high"
        elif avg_confidence >= 0.8:
            return "high"
        elif avg_confidence >= 0.7:
            return "medium"
        else:
            return "low"

    def _is_sql_ready(self, tables, intents):
        """Check if extraction is ready for SQL generation"""
        # Need at least one table and one intent
        has_table = len(tables) > 0
        has_intent = len(intents) > 0
        
        # Check confidence levels
        table_confidence_ok = len(tables) == 0 or tables[0]["confidence"] >= 0.7
        intent_confidence_ok = len(intents) == 0 or intents[0]["confidence"] >= 0.7
        
        return has_table and has_intent and table_confidence_ok and intent_confidence_ok

    def get_extraction_summary(self, text):
        """Get extraction summary (compatibility method)"""
        extraction = self.extract(text)
        
        return {
            "input_text": text,
            "tables_detected": len(extraction["tables"]),
            "time_filters_detected": len(extraction["time_filters"]),
            "intents_detected": len(extraction["intents"]),
            "complexity": extraction["metadata"]["complexity"],
            "confidence_level": extraction["metadata"]["confidence_level"],
            "top_table": extraction["tables"][0]["table"] if extraction["tables"] else None,
            "top_time_filter": extraction["time_filters"][0]["period"] if extraction["time_filters"] else None,
            "primary_intent": extraction["primary_intent"]["type"] if extraction["primary_intent"] else None,
            "sql_ready": extraction["metadata"]["sql_ready"]
        }

    def get_statistics(self):
        """Get extraction statistics"""
        success_rate = (self.successful_extractions / self.extracted_queries * 100) if self.extracted_queries > 0 else 0
        
        return {
            "total_extractions": self.extracted_queries,
            "successful_extractions": self.successful_extractions,
            "success_rate": round(success_rate, 2),
            "extraction_method": "trained_ner_model",
            "model_loaded": self.is_loaded,
            "model_info": self.ner_model.get_model_info() if self.is_loaded else None
        }

    def is_ready(self):
        """Check if extractor is ready for use"""
        return self.is_loaded


def create_entity_extractor():
    """Factory function to create entity extractor"""
    return EntityExtractor()


# Test function
if __name__ == "__main__":
    print("🧪 Testing NER-based Entity Extractor...")
    
    extractor = EntityExtractor()
    
    if extractor.is_ready():
        test_texts = [
            "Bu ayın müşteri sayısı",
            "Geçen yıl sipariş toplamı",
            "Son 30 gün ürün listesi", 
            "2022 3. çeyrek çalışan maaşları"
        ]
        
        for text in test_texts:
            print(f"\n🔍 Testing: '{text}'")
            
            try:
                result = extractor.extract(text)
                
                print(f"📊 Summary:")
                print(f"  Tables: {len(result['tables'])}")
                print(f"  Time filters: {len(result['time_filters'])}")
                print(f"  Intents: {len(result['intents'])}")
                print(f"  Primary intent: {result['primary_intent']['type'] if result['primary_intent'] else 'None'}")
                print(f"  SQL ready: {result['metadata']['sql_ready']}")
                print(f"  Confidence: {result['metadata']['confidence_level']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Show statistics
        stats = extractor.get_statistics()
        print(f"\n📈 Statistics: {stats}")
        
    else:
        print("❌ Entity extractor not ready!")