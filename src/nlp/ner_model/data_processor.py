# src/nlp/ner_model/data_processor.py
"""
Data Processor for Turkish NER Model
Converts NER JSON data to training format for BERT-based model
"""

import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from config.model_config import BERTURK_MODEL_NAME


class NERDataProcessor:
    """
    Processes NER training data for Turkish BERT-based model
    Handles tokenization, label encoding, and data formatting
    """

    def __init__(self, tokenizer_name=BERTURK_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Special tokens
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token

        # Label mappings
        self.label_to_id = {}
        self.id_to_label = {}
        self._create_label_mappings()

        # Statistics
        self.processed_samples = 0
        self.tokenization_errors = 0

    def _create_label_mappings(self):
        """Create comprehensive label mappings for all entity types"""

        # All entity labels from our NER generator
        base_labels = [
            # Table entities
            "TABLE_CUSTOMERS", "TABLE_PRODUCTS", "TABLE_ORDERS",
            "TABLE_CATEGORIES", "TABLE_SUPPLIERS", "TABLE_EMPLOYEES",
            "TABLE_ORDER_DETAILS", "TABLE_PURCHASE_ORDERS",

            # Intent entities
            "INTENT_SELECT", "INTENT_COUNT", "INTENT_SUM", "INTENT_AVG",

            # Basic time entities
            "TIME_CURRENT_MONTH", "TIME_CURRENT_YEAR", "TIME_LAST_MONTH",
            "TIME_LAST_YEAR", "TIME_TODAY", "TIME_CURRENT_WEEK", "TIME_LAST_WEEK",

            # Advanced time entities
            "TIME_LAST_N_DAYS", "TIME_LAST_N_WEEKS", "TIME_LAST_N_MONTHS",
            "TIME_LAST_N_YEARS", "TIME_NUMBER", "TIME_UNIT",

            # Quarter entities
            "TIME_Q1", "TIME_Q2", "TIME_Q3", "TIME_Q4",
            "TIME_CURRENT_QUARTER", "TIME_LAST_QUARTER",

            # Specific dates and ranges
            "TIME_SPECIFIC_DATE", "TIME_RANGE_START", "TIME_RANGE_END",

            # Action verbs
            "ACTION_VERB"
        ]

        # Create BIO tags for each label
        all_labels = ["O"]  # Outside tag

        for label in base_labels:
            all_labels.extend([f"B-{label}", f"I-{label}"])

        # Create mappings
        self.label_to_id = {label: idx for idx, label in enumerate(all_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        print(f"📋 Created label mappings: {len(all_labels)} labels")

    def load_ner_data(self, file_path):
        """Load NER training data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            training_data = data.get("training_data", [])
            meta_info = data.get("meta", {})

            print(f"📁 Loaded NER data: {len(training_data)} samples")
            print(f"📊 Meta info: {meta_info.get('total_samples', 'Unknown')} total samples")

            return training_data, meta_info

        except Exception as e:
            print(f"❌ Error loading NER data: {e}")
            return [], {}

    def convert_to_bio_format(self, training_data):
        """Convert entity annotations to BIO format"""
        bio_data = []

        for sample in training_data:
            text = sample["text"]
            entities = sample.get("entities", [])

            # Tokenize text
            tokens = text.split()  # Simple whitespace tokenization

            # Initialize all tokens as 'O' (Outside)
            bio_labels = ["O"] * len(tokens)

            # Process entities
            for entity in entities:
                entity_text = entity["text"]
                entity_label = entity["label"]

                # Find token positions for this entity
                entity_tokens = entity_text.split()

                # Find where this entity appears in the token list
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[i:i + len(entity_tokens)] == entity_tokens:
                        # Mark first token as B- (Beginning)
                        bio_labels[i] = f"B-{entity_label}"

                        # Mark remaining tokens as I- (Inside)
                        for j in range(1, len(entity_tokens)):
                            if i + j < len(bio_labels):
                                bio_labels[i + j] = f"I-{entity_label}"
                        break

            bio_data.append({
                "text": text,
                "tokens": tokens,
                "labels": bio_labels
            })

        print(f"🏷️ Converted {len(bio_data)} samples to BIO format")
        return bio_data

    def tokenize_and_align(self, bio_data, max_length=512):
        """Tokenize with BERT tokenizer and align labels"""
        tokenized_data = []

        for sample in bio_data:
            tokens = sample["tokens"]
            labels = sample["labels"]

            # BERT tokenization
            tokenized_inputs = self.tokenizer(
                tokens,
                is_split_into_words=True,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )

            # Align labels with BERT tokens
            word_ids = tokenized_inputs.word_ids()
            aligned_labels = []

            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens ([CLS], [SEP], [PAD])
                    aligned_labels.append("O")
                elif word_idx != previous_word_idx:
                    # First subtoken of a word
                    if word_idx < len(labels):
                        aligned_labels.append(labels[word_idx])
                    else:
                        aligned_labels.append("O")
                else:
                    # Subsequent subtokens of the same word
                    if word_idx < len(labels):
                        label = labels[word_idx]
                        # Convert B- to I- for subtokens
                        if label.startswith("B-"):
                            aligned_labels.append(label.replace("B-", "I-"))
                        else:
                            aligned_labels.append(label)
                    else:
                        aligned_labels.append("O")

                previous_word_idx = word_idx

            # Convert labels to IDs
            label_ids = [self.label_to_id.get(label, 0) for label in aligned_labels]

            tokenized_data.append({
                "input_ids": tokenized_inputs["input_ids"],
                "attention_mask": tokenized_inputs["attention_mask"],
                "labels": label_ids,
                "tokens": self.tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"]),
                "original_text": sample["text"]
            })

            self.processed_samples += 1

        print(f"🔤 Tokenized {len(tokenized_data)} samples with BERT tokenizer")
        return tokenized_data

    def create_dataset_splits(self, tokenized_data, train_ratio=0.8, val_ratio=0.1):
        """Split data into train/validation/test sets"""
        import random

        # Shuffle data
        data_copy = tokenized_data.copy()
        random.shuffle(data_copy)

        total_samples = len(data_copy)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)

        train_data = data_copy[:train_size]
        val_data = data_copy[train_size:train_size + val_size]
        test_data = data_copy[train_size + val_size:]

        print(f"📊 Dataset splits:")
        print(f"   Train: {len(train_data)} samples ({train_ratio * 100:.1f}%)")
        print(f"   Validation: {len(val_data)} samples ({val_ratio * 100:.1f}%)")
        print(f"   Test: {len(test_data)} samples ({(1 - train_ratio - val_ratio) * 100:.1f}%)")

        return train_data, val_data, test_data

    def process_ner_data(self, file_path, max_length=512):
        """Complete processing pipeline"""
        print("🚀 Starting NER data processing pipeline...")

        # Step 1: Load data
        training_data, meta_info = self.load_ner_data(file_path)
        if not training_data:
            return None

        # Step 2: Convert to BIO format
        bio_data = self.convert_to_bio_format(training_data)

        # Step 3: Tokenize and align
        tokenized_data = self.tokenize_and_align(bio_data, max_length)

        # Step 4: Create splits
        train_data, val_data, test_data = self.create_dataset_splits(tokenized_data)

        # Step 5: Statistics
        self._print_processing_stats(tokenized_data, meta_info)

        return {
            "train": train_data,
            "validation": val_data,
            "test": test_data,
            "label_mappings": {
                "label_to_id": self.label_to_id,
                "id_to_label": self.id_to_label
            },
            "meta": meta_info
        }

    def _print_processing_stats(self, tokenized_data, meta_info):
        """Print processing statistics"""
        print(f"\n📈 Processing Statistics:")
        print(f"   Total processed samples: {self.processed_samples}")
        print(f"   Tokenization errors: {self.tokenization_errors}")
        print(f"   Average tokens per sample: {np.mean([len(sample['input_ids']) for sample in tokenized_data]):.1f}")
        print(f"   Total unique labels: {len(self.label_to_id)}")

        # Label distribution
        label_counts = {}
        for sample in tokenized_data:
            for label_id in sample["labels"]:
                label = self.id_to_label[label_id]
                label_counts[label] = label_counts.get(label, 0) + 1

        print(f"\n🏷️ Top 10 Most Common Labels:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {label}: {count}")

    def save_processed_data(self, processed_data, output_dir="../../../models/ner_data"):
        """Save processed data for training"""
        import os
        import pickle

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save each split
        for split_name, data in processed_data.items():
            if split_name in ["train", "validation", "test"]:
                output_file = os.path.join(output_dir, f"{split_name}_data.pkl")
                with open(output_file, 'wb') as f:
                    pickle.dump(data, f)
                print(f"💾 Saved {split_name} data: {len(data)} samples")

        # Save label mappings
        mappings_file = os.path.join(output_dir, "label_mappings.json")
        with open(mappings_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data["label_mappings"], f, ensure_ascii=False, indent=2)
        print(f"💾 Saved label mappings: {len(processed_data['label_mappings']['label_to_id'])} labels")

        return output_dir

    def get_label_info(self):
        """Get label information"""
        return {
            "total_labels": len(self.label_to_id),
            "label_to_id": self.label_to_id,
            "id_to_label": self.id_to_label,
            "entity_types": [label.split('-')[1] for label in self.label_to_id.keys() if '-' in label]
        }


class NERDataset(Dataset):
    """PyTorch Dataset for NER training"""

    def __init__(self, data, max_length=512):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Pad sequences to max_length
        input_ids = item["input_ids"][:self.max_length]
        attention_mask = item["attention_mask"][:self.max_length]
        labels = item["labels"][:self.max_length]

        # Pad if necessary
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([0] * padding_length)  # PAD token ID
            attention_mask.extend([0] * padding_length)
            labels.extend([-100] * padding_length)  # Ignore index for loss

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def create_ner_data_processor():
    """Factory function to create NER data processor"""
    return NERDataProcessor()


# Test/Demo function
if __name__ == "__main__":
    # Test the data processor
    processor = NERDataProcessor()

    # Process the generated NER data
    ner_data_path = "../../../data/ner_training_data.json"  # Adjust path as needed

    try:
        processed_data = processor.process_ner_data(ner_data_path)

        if processed_data:
            # Save processed data
            output_dir = processor.save_processed_data(processed_data)
            print(f"\n✅ Data processing complete! Output: {output_dir}")

            # Show sample
            sample = processed_data["train"][0]
            print(f"\n🔍 Sample processed data:")
            print(f"   Text: '{sample['original_text']}'")
            print(f"   Tokens: {len(sample['input_ids'])}")
            print(f"   Labels: {len(sample['labels'])}")

        else:
            print("❌ Data processing failed!")

    except Exception as e:
        print(f"❌ Error: {e}")