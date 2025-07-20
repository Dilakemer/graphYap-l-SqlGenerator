# src/nlp/ner_model/ner_trainer.py
"""
NER Trainer for Turkish NLP-SQL Project
Handles training, validation, and evaluation of Turkish NER model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pickle
import json
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from config.model_config import MODELS_DIR

# Import our modules
from data_processor import NERDataset
from turkish_ner import TurkishNER


class NERTrainer:
    """
    Trainer for Turkish NER Model
    Handles complete training pipeline with validation and evaluation
    """

    def __init__(self, model_name="dbmdz/bert-base-turkish-cased"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None

        # Training configuration
        self.config = {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 10,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "early_stopping_patience": 3,
            "save_best_model": True
        }

        # Training state
        self.current_epoch = 0
        self.best_f1_score = 0.0
        self.training_history = []
        self.early_stopping_counter = 0

        # Paths
        self.data_dir = MODELS_DIR / "ner_data"
        self.model_save_dir = MODELS_DIR / "ner_model"

        print(f"🎯 NER Trainer initialized")
        print(f"📱 Device: {self.device}")
        print(f"⚙️ Configuration: {self.config}")

    def load_data(self, data_dir=None):
        """Load processed training data"""
        if data_dir is None:
            data_dir = self.data_dir

        try:
            # Load training data
            with open(data_dir / "train_data.pkl", 'rb') as f:
                train_data = pickle.load(f)

            with open(data_dir / "validation_data.pkl", 'rb') as f:
                val_data = pickle.load(f)

            with open(data_dir / "test_data.pkl", 'rb') as f:
                test_data = pickle.load(f)

            # Load label mappings
            with open(data_dir / "label_mappings.json", 'r', encoding='utf-8') as f:
                label_mappings = json.load(f)

            # Create datasets
            train_dataset = NERDataset(train_data, max_length=512)
            val_dataset = NERDataset(val_data, max_length=512)
            test_dataset = NERDataset(test_data, max_length=512)

            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=0  # Set to 0 for Windows compatibility
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=0
            )

            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=0
            )

            print(f"📊 Data loaded successfully:")
            print(f"   Train: {len(train_data)} samples")
            print(f"   Validation: {len(val_data)} samples")
            print(f"   Test: {len(test_data)} samples")
            print(f"   Labels: {len(label_mappings['label_to_id'])} unique labels")

            return label_mappings

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def initialize_model(self, label_mappings):
        """Initialize model and training components"""
        try:
            # Create Turkish NER model
            self.model = TurkishNER(model_name=self.model_name)

            # Set label mappings manually
            self.model.label_to_id = label_mappings["label_to_id"]
            self.model.id_to_label = {int(k): v for k, v in label_mappings["id_to_label"].items()}
            self.model.num_labels = len(label_mappings["label_to_id"])

            # Initialize model
            if not self.model.initialize_model():
                return False

            # Setup optimizer
            self.optimizer = AdamW(
                self.model.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )

            # Calculate total training steps
            total_steps = len(self.train_loader) * self.config["num_epochs"]
            warmup_steps = int(total_steps * self.config["warmup_ratio"])

            # Setup scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            print(f"🎯 Model and training components initialized")
            print(f"📈 Total training steps: {total_steps}")
            print(f"🔥 Warmup steps: {warmup_steps}")

            return True

        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            return False

    def train_epoch(self):
        """Train for one epoch"""
        self.model.model.train()
        total_loss = 0
        predictions = []
        true_labels = []

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(),
                self.config["max_grad_norm"]
            )

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            total_loss += loss.item()

            # Get predictions for metrics
            batch_predictions = torch.argmax(logits, dim=-1)

            # Flatten and filter out padding (-100 labels)
            for i in range(labels.size(0)):
                label_mask = labels[i] != -100
                true_batch_labels = labels[i][label_mask].cpu().numpy()
                pred_batch_labels = batch_predictions[i][label_mask].cpu().numpy()

                true_labels.extend(true_batch_labels)
                predictions.extend(pred_batch_labels)

            # Print progress
            if batch_idx % 50 == 0:
                print(f"    Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        f1_score = self._calculate_f1_score(true_labels, predictions)

        return avg_loss, f1_score

    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                # Get predictions
                batch_predictions = torch.argmax(logits, dim=-1)

                # Flatten and filter out padding
                for i in range(labels.size(0)):
                    label_mask = labels[i] != -100
                    true_batch_labels = labels[i][label_mask].cpu().numpy()
                    pred_batch_labels = batch_predictions[i][label_mask].cpu().numpy()

                    true_labels.extend(true_batch_labels)
                    predictions.extend(pred_batch_labels)

        # Calculate validation metrics
        avg_loss = total_loss / len(self.val_loader)
        f1_score = self._calculate_f1_score(true_labels, predictions)
        precision = self._calculate_precision(true_labels, predictions)
        recall = self._calculate_recall(true_labels, predictions)

        return avg_loss, f1_score, precision, recall

    def _calculate_f1_score(self, true_labels, predictions):
        """Calculate macro F1 score"""
        from sklearn.metrics import f1_score
        return f1_score(true_labels, predictions, average='macro', zero_division=0)

    def _calculate_precision(self, true_labels, predictions):
        """Calculate macro precision"""
        from sklearn.metrics import precision_score
        return precision_score(true_labels, predictions, average='macro', zero_division=0)

    def _calculate_recall(self, true_labels, predictions):
        """Calculate macro recall"""
        from sklearn.metrics import recall_score
        return recall_score(true_labels, predictions, average='macro', zero_division=0)

    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config["num_epochs"]

        print(f"🚀 Starting training for {num_epochs} epochs...")
        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_start_time = time.time()

            print(f"\n📈 Epoch {self.current_epoch}/{num_epochs}")
            print("-" * 50)

            # Training
            train_loss, train_f1 = self.train_epoch()

            # Validation
            val_loss, val_f1, val_precision, val_recall = self.validate_epoch()

            epoch_time = time.time() - epoch_start_time

            # Log metrics
            epoch_metrics = {
                "epoch": self.current_epoch,
                "train_loss": train_loss,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "epoch_time": epoch_time,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }

            self.training_history.append(epoch_metrics)

            # Print epoch results
            print(f"⏱️ Epoch {self.current_epoch} completed in {epoch_time:.2f}s")
            print(f"📊 Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            print(f"📊 Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            print(f"📊 Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")

            # Save best model
            if val_f1 > self.best_f1_score:
                self.best_f1_score = val_f1
                self.early_stopping_counter = 0

                if self.config["save_best_model"]:
                    self.save_checkpoint(is_best=True)
                    print(f"💾 New best model saved! F1: {val_f1:.4f}")
            else:
                self.early_stopping_counter += 1

            # Early stopping check
            if self.early_stopping_counter >= self.config["early_stopping_patience"]:
                print(f"⏹️ Early stopping triggered after {self.current_epoch} epochs")
                break

            # Save regular checkpoint
            if self.current_epoch % 2 == 0:  # Save every 2 epochs
                self.save_checkpoint(is_best=False)

        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.2f} seconds")
        print(f"🏆 Best F1 Score: {self.best_f1_score:.4f}")

        # Save training history
        self.save_training_history()

        return self.training_history

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        try:
            checkpoint_dir = self.model_save_dir
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            if is_best:
                # Save best model
                self.model.save_model(checkpoint_dir / "best_model")

                # Save training state
                state = {
                    "epoch": self.current_epoch,
                    "best_f1_score": self.best_f1_score,
                    "config": self.config,
                    "training_history": self.training_history
                }

                torch.save(state, checkpoint_dir / "best_training_state.pt")

            else:
                # Save regular checkpoint
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"

                state = {
                    "epoch": self.current_epoch,
                    "model_state_dict": self.model.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_f1_score": self.best_f1_score,
                    "config": self.config
                }

                torch.save(state, checkpoint_path)

            return True

        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            return False

    def save_training_history(self):
        """Save training history to JSON"""
        try:
            history_path = self.model_save_dir / "training_history.json"

            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2)

            print(f"📊 Training history saved to: {history_path}")
            return True

        except Exception as e:
            print(f"❌ Error saving training history: {e}")
            return False

    def evaluate_on_test(self):
        """Evaluate model on test set"""
        print(f"\n🧪 Evaluating on test set...")

        self.model.model.eval()
        all_predictions = []
        all_true_labels = []
        test_loss = 0

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                test_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)

                # Collect predictions and labels
                for i in range(labels.size(0)):
                    label_mask = labels[i] != -100
                    true_batch_labels = labels[i][label_mask].cpu().numpy()
                    pred_batch_labels = predictions[i][label_mask].cpu().numpy()

                    all_true_labels.extend(true_batch_labels)
                    all_predictions.extend(pred_batch_labels)

        # Calculate test metrics
        test_f1 = self._calculate_f1_score(all_true_labels, all_predictions)
        test_precision = self._calculate_precision(all_true_labels, all_predictions)
        test_recall = self._calculate_recall(all_true_labels, all_predictions)
        avg_test_loss = test_loss / len(self.test_loader)

        test_results = {
            "test_loss": avg_test_loss,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "total_predictions": len(all_predictions)
        }

        print(f"📊 Test Results:")
        print(f"   Loss: {avg_test_loss:.4f}")
        print(f"   F1 Score: {test_f1:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall: {test_recall:.4f}")

        return test_results

    def run_full_training(self):
        """Complete training pipeline"""
        print("🚀 Starting full NER training pipeline...")

        # Step 1: Load data
        label_mappings = self.load_data()
        if not label_mappings:
            print("❌ Failed to load data")
            return False

        # Step 2: Initialize model
        if not self.initialize_model(label_mappings):
            print("❌ Failed to initialize model")
            return False

        # Step 3: Train model
        training_history = self.train()

        # Step 4: Evaluate on test set
        test_results = self.evaluate_on_test()

        # Step 5: Save final results
        final_results = {
            "training_completed": True,
            "best_f1_score": self.best_f1_score,
            "test_results": test_results,
            "training_config": self.config,
            "model_path": str(self.model_save_dir)
        }

        with open(self.model_save_dir / "final_results.json", 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"🏆 Best validation F1: {self.best_f1_score:.4f}")
        print(f"🧪 Test F1: {test_results['test_f1']:.4f}")
        print(f"💾 Model saved to: {self.model_save_dir}")

        return True


def create_ner_trainer():
    """Factory function to create NER trainer"""
    return NERTrainer()


# Training script
if __name__ == "__main__":
    print("🎯 Turkish NER Model Training")
    print("=" * 50)

    # Create trainer
    trainer = NERTrainer()

    # Run full training pipeline
    success = trainer.run_full_training()

    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")