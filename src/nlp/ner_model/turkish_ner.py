# src/nlp/ner_model/turkish_ner.py
"""
Turkish NER Model for NLP-SQL Project
BERT-based Named Entity Recognition for Turkish text
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig
)
import json
import pickle
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from config.model_config import BERTURK_MODEL_NAME, MODELS_DIR


class TurkishNER:
    """
    Turkish Named Entity Recognition Model
    Uses BERTurk with Token Classification head for entity detection
    """

    def __init__(self, model_name=BERTURK_MODEL_NAME, num_labels=None):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model components
        self.tokenizer = None
        self.model = None
        self.config = None

        # Label mappings
        self.label_to_id = {}
        self.id_to_label = {}
        self.num_labels = num_labels

        # Model state
        self.is_trained = False
        self.model_path = MODELS_DIR / "ner_model"

        print(f"🤖 Turkish NER Model initialized")
        print(f"📱 Device: {self.device}")

    def load_label_mappings(self, mappings_path):
        """Load label mappings from JSON file"""
        try:
            with open(mappings_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)

            self.label_to_id = mappings["label_to_id"]
            self.id_to_label = {int(k): v for k, v in mappings["id_to_label"].items()}
            self.num_labels = len(self.label_to_id)

            print(f"📋 Loaded label mappings: {self.num_labels} labels")
            return True

        except Exception as e:
            print(f"❌ Error loading label mappings: {e}")
            return False

    def initialize_model(self, mappings_path=None):
        """Initialize tokenizer and model"""
        try:
            # Load label mappings first
            if mappings_path:
                if not self.load_label_mappings(mappings_path):
                    return False
            elif not self.label_to_id:
                # Try default path
                default_mappings = MODELS_DIR / "ner_data" / "label_mappings.json"
                if not self.load_label_mappings(default_mappings):
                    print("❌ No label mappings found!")
                    return False

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print(f"🔤 Tokenizer loaded: {self.model_name}")

            # Initialize model configuration
            self.config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                id2label=self.id_to_label,
                label2id=self.label_to_id
            )

            # Initialize model
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                config=self.config
            )

            # Move to device
            self.model.to(self.device)

            print(f"🎯 Model initialized with {self.num_labels} labels")
            return True

        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            return False

    def predict(self, text, return_confidence=False):
        """
        Predict entities in text

        Args:
            text: Input text string
            return_confidence: Whether to return confidence scores

        Returns:
            List of detected entities
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        offset_mapping = inputs["offset_mapping"]

        # Model prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_ids = torch.argmax(predictions, dim=-1)

        # Convert predictions to entities
        entities = self._extract_entities_from_predictions(
            text,
            predicted_ids[0].cpu().numpy(),
            offset_mapping[0],
            predictions[0].cpu().numpy() if return_confidence else None
        )

        return entities

    def predict_batch(self, texts, batch_size=8):
        """Predict entities for multiple texts"""
        all_entities = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_entities = []

            for text in batch_texts:
                entities = self.predict(text)
                batch_entities.append(entities)

            all_entities.extend(batch_entities)

        return all_entities

    def _extract_entities_from_predictions(self, text, predicted_ids, offset_mapping, confidence_scores=None):
        """Extract entities from model predictions"""
        entities = []
        current_entity = None

        for i, (pred_id, (start, end)) in enumerate(zip(predicted_ids, offset_mapping)):
            if start == 0 and end == 0:  # Special tokens
                continue

            predicted_label = self.id_to_label.get(pred_id, "O")
            confidence = float(np.max(confidence_scores[i])) if confidence_scores is not None else None

            if predicted_label == "O":
                # End current entity if exists
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

            elif predicted_label.startswith("B-"):
                # Begin new entity
                if current_entity:
                    entities.append(current_entity)

                entity_type = predicted_label[2:]  # Remove "B-" prefix
                current_entity = {
                    "text": text[start:end],
                    "label": entity_type,
                    "start": start,
                    "end": end,
                    "confidence": confidence
                }

            elif predicted_label.startswith("I-") and current_entity:
                # Continue current entity
                entity_type = predicted_label[2:]  # Remove "I-" prefix

                if current_entity["label"] == entity_type:
                    # Extend current entity
                    current_entity["end"] = end
                    current_entity["text"] = text[current_entity["start"]:end]

                    # Update confidence (average)
                    if confidence and current_entity["confidence"]:
                        current_entity["confidence"] = (current_entity["confidence"] + confidence) / 2
                else:
                    # Label mismatch, start new entity
                    entities.append(current_entity)
                    current_entity = {
                        "text": text[start:end],
                        "label": entity_type,
                        "start": start,
                        "end": end,
                        "confidence": confidence
                    }

        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)

        return entities

    def extract_tables_and_times(self, text):
        """
        Extract tables and time entities specifically
        Compatible with existing entity_extractor interface
        """
        entities = self.predict(text, return_confidence=True)

        # Separate by entity type
        tables = []
        time_filters = []

        for entity in entities:
            label = entity["label"]

            if label.startswith("TABLE_"):
                # Map to original table names
                table_name = self._map_table_label_to_name(label)
                tables.append({
                    "table": table_name,
                    "confidence": entity["confidence"],
                    "matched_pattern": entity["text"],
                    "start": entity["start"],
                    "end": entity["end"]
                })

            elif label.startswith("TIME_"):
                # Map to time period
                time_period = self._map_time_label_to_period(label, entity["text"])
                time_filters.append({
                    "period": time_period,
                    "confidence": entity["confidence"],
                    "matched_pattern": entity["text"],
                    "start": entity["start"],
                    "end": entity["end"]
                })

        # Sort by confidence
        tables.sort(key=lambda x: x["confidence"] if x["confidence"] else 0, reverse=True)
        time_filters.sort(key=lambda x: x["confidence"] if x["confidence"] else 0, reverse=True)

        return {
            "tables": tables,
            "time_filters": time_filters,
            "all_entities": entities
        }

    def _map_table_label_to_name(self, label):
        """Map TABLE_ labels to table names"""
        mapping = {
            "TABLE_CUSTOMERS": "customers",
            "TABLE_PRODUCTS": "products",
            "TABLE_ORDERS": "orders",
            "TABLE_CATEGORIES": "categories",
            "TABLE_SUPPLIERS": "suppliers",
            "TABLE_EMPLOYEES": "employees",
            "TABLE_ORDER_DETAILS": "order_details",
            "TABLE_PURCHASE_ORDERS": "purchase_orders"
        }
        return mapping.get(label, label.lower().replace("table_", ""))

    def _map_time_label_to_period(self, label, text):
        """Map TIME_ labels to time periods"""
        # Basic mappings
        basic_mapping = {
            "TIME_CURRENT_MONTH": "current_month",
            "TIME_CURRENT_YEAR": "current_year",
            "TIME_LAST_MONTH": "last_month",
            "TIME_LAST_YEAR": "last_year",
            "TIME_TODAY": "today",
            "TIME_CURRENT_WEEK": "current_week",
            "TIME_LAST_WEEK": "last_week",
            "TIME_CURRENT_QUARTER": "current_quarter",
            "TIME_LAST_QUARTER": "last_quarter",
            "TIME_Q1": "q1",
            "TIME_Q2": "q2",
            "TIME_Q3": "q3",
            "TIME_Q4": "q4"
        }

        if label in basic_mapping:
            return basic_mapping[label]

        # Handle relative time periods
        if label.startswith("TIME_LAST_N_"):
            return f"last_n_{label.split('_')[-1].lower()}"

        # Handle specific dates
        if label == "TIME_SPECIFIC_DATE":
            return f"specific_date:{text}"

        # Default fallback
        return label.lower().replace("time_", "")

    def save_model(self, save_path=None):
        """Save trained model"""
        if save_path is None:
            save_path = self.model_path

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save model and tokenizer
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

            # Save label mappings
            mappings = {
                "label_to_id": self.label_to_id,
                "id_to_label": self.id_to_label,
                "num_labels": self.num_labels
            }

            with open(save_path / "label_mappings.json", 'w', encoding='utf-8') as f:
                json.dump(mappings, f, ensure_ascii=False, indent=2)

            print(f"💾 Model saved to: {save_path}")
            return True

        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

    def load_model(self, load_path=None):
        """Load trained model"""
        if load_path is None:
            # Try best_model first (from training)
            best_model_path = self.model_path / "best_model"
            if (best_model_path / "config.json").exists():
                load_path = best_model_path
            else:
                load_path = self.model_path

        load_path = Path(load_path)

        try:
            # Check if model exists
            if not (load_path / "config.json").exists():
                print(f"⚠️ No trained model found at: {load_path}")
                return False

            # Load label mappings
            mappings_path = load_path / "label_mappings.json"
            if not self.load_label_mappings(mappings_path):
                return False

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)

            # Load model
            self.model = AutoModelForTokenClassification.from_pretrained(load_path)
            self.model.to(self.device)

            self.is_trained = True
            print(f"📥 Model loaded from: {load_path}")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def get_model_info(self):
        """Get model information"""
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "device": str(self.device),
            "is_trained": self.is_trained,
            "model_path": str(self.model_path),
            "supported_entities": list(self.label_to_id.keys()) if self.label_to_id else []
        }

    def evaluate_on_text(self, text, expected_entities=None):
        """Evaluate model performance on single text"""
        predicted_entities = self.predict(text, return_confidence=True)

        result = {
            "text": text,
            "predicted_entities": predicted_entities,
            "entity_count": len(predicted_entities)
        }

        if expected_entities:
            # Calculate simple precision/recall
            predicted_labels = {(e["start"], e["end"], e["label"]) for e in predicted_entities}
            expected_labels = {(e["start"], e["end"], e["label"]) for e in expected_entities}

            true_positives = len(predicted_labels & expected_labels)
            precision = true_positives / len(predicted_labels) if predicted_labels else 0
            recall = true_positives / len(expected_labels) if expected_labels else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            result.update({
                "expected_entities": expected_entities,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

        return result


def create_turkish_ner():
    """Factory function to create Turkish NER model"""
    return TurkishNER()


def main():
    """Main function for testing"""
    print("🚀 Testing Turkish NER Model...")

    # Create model instance
    ner_model = TurkishNER()

    # Try to load trained model first
    print("📥 Attempting to load trained model...")
    if ner_model.load_model():
        print("✅ Trained model loaded successfully!")
        model_status = "trained"
    else:
        print("⚠️ No trained model found, initializing fresh model...")
        if ner_model.initialize_model():
            print("✅ Fresh model initialized!")
            model_status = "untrained"
        else:
            print("❌ Model initialization failed!")
            return

    # Show model info
    info = ner_model.get_model_info()
    print(f"📊 Model Info: {info['num_labels']} labels, Device: {info['device']}, Trained: {info['is_trained']}")

    if ner_model.model:
        # Test prediction with confidence
        test_texts = [
            "Bu ayın müşteri sayısı",
            "Geçen yıl sipariş toplamı",
            "Son 30 gün ürün listesi",
            "2022 3. çeyrek çalışan maaşları"
        ]

        print(f"\n🔍 Testing prediction on sample texts ({model_status} model):")
        for text in test_texts:
            try:
                entities = ner_model.predict(text, return_confidence=True)
                print(f"\n  Text: '{text}'")
                print(f"  Entities: {len(entities)}")

                if len(entities) == 0:
                    print("    No entities detected")
                else:
                    for entity in entities:
                        confidence = entity.get('confidence', 'N/A')
                        if confidence != 'N/A' and confidence is not None:
                            confidence = f"{confidence:.3f}"
                        print(f"    - '{entity['text']}' → {entity['label']} (confidence: {confidence})")

            except Exception as e:
                print(f"  ❌ Error predicting '{text}': {e}")

        # Test table/time extraction
        print(f"\n🎯 Testing table/time extraction:")
        test_text = "Bu ayın müşteri sayısını göster"
        try:
            result = ner_model.extract_tables_and_times(test_text)
            print(f"  Text: '{test_text}'")
            print(f"  Tables detected: {len(result['tables'])}")
            for table in result['tables']:
                print(
                    f"    - {table['table']} (confidence: {table['confidence']:.3f if table['confidence'] else 'N/A'})")

            print(f"  Time filters detected: {len(result['time_filters'])}")
            for time_filter in result['time_filters']:
                print(
                    f"    - {time_filter['period']} (confidence: {time_filter['confidence']:.3f if time_filter['confidence'] else 'N/A'})")

        except Exception as e:
            print(f"  ❌ Error in table/time extraction: {e}")

    else:
        print("❌ Model not available for testing!")


# Entry point
if __name__ == "__main__":
    main()