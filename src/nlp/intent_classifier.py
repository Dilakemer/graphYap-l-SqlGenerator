"""
Intent Classifier - Turkish NLP Intent Detection
Uses BERTurk embeddings + Sklearn for intent classification

In order to understand the essentials about Logistic Regression:
    https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe
"""

from .berturk_wrapper import BERTurkWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import json
import os
import joblib


class IntentClassifier:
    """
    Intent classification for Turkish SQL queries
    Supports: SELECT, COUNT, SUM, AVG intents
    """

    def __init__(self):
        """Initialize intent classifier"""
        self.berturk = BERTurkWrapper()
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.intent_labels = ["SELECT", "COUNT", "SUM", "AVG"]
        self.accuracy = 0
        self.is_trained = False
        self.model_path = "models/sklearn_models/intent_classifier.pkl"

        # Create models directory if not exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def prepare_training_data(self, training_file):
        """
        Load and prepare training data from JSON

        Args:
            training_file: the path to JSON training data(in our case data/intent_training_data.json)
        Returns:
            X: embeddings array, y: intent labels array
        """

        try:
            with open(training_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            texts = []
            labels = []

            # Extract texts and labels from JSON
            if "training_data" in data:
                for item in data["training_data"]:
                    texts.append(item["text"])
                    labels.append(item["intent"])
            else:
                # Should be direct format without nesting
                for item in data:
                    texts.append(item["text"])
                    labels.append(item["intent"])

            # Generate embeddings via BERTurk
            X = self.berturk.get_embeddings_batch(texts)
            y = np.array(labels)

            return X, y
        except Exception as e:
            raise RuntimeError(f"Error while preparing training data: {e}")

    def train(self, training_file, test_size=0.2):
        """
        Train intent classifier with Turkish training data

        Args:
            training_file: Path to JSON training data
            test_size: Fraction of data for testing
        """

        try:
            # Prepare training data
            X, y = self.prepare_training_data(training_file)

            # In the JSON data, the distribution of the number of intents are different.
            # For example: 40 select, 25 sum, 20 count, 15 avg. Thus, it is very likely that train_test_split
            # will choose more select than its real proportion. stratify=y makes the splitting by keeping the
            # original proportion same as original training data source
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Train classifier
            self.classifier.fit(X_train, y_train)

            # Yield predictions for y by using X_test
            y_pred = self.classifier.predict(X_test)
            # Measure the accuracy.
            # We dont want either overfitting(aşırı öğrenme sonucunda tahminin taklite dönüşmesi olarak çevrilebilir)
            self.accuracy = accuracy_score(y_test, y_pred)

            self.is_trained = True

            # Save model
            self.save_model()

        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")

    def predict(self, text):
        """
        Predict intent for Turkish text
            Args: turkish text input
        Returns:
            Dictionary with intent and confidence
        """

        if not self.is_trained:
            # Try to load the saved model
            if not self.load_model():
                raise RuntimeError("Model not trained. Call train() first or provide training data")

        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        try:
            # Generate embedding
            embedding = self.berturk.get_embeddings(text)

            # Predict intent ('SELECT', 'SUM' etc.)
            intent_pred = self.classifier.predict([embedding])[0]

            # Get confidence scores
            confidence_scores = self.classifier.predict_proba([embedding])[0]
            max_confidence = np.max(confidence_scores)

            # Get confidence for each class
            confidence_by_class = {}

            # confidence scores is similar to -> [0.25, 0.8, 0.03, 0.02]
            for i, intent in enumerate(self.classifier.classes_):
                confidence_by_class[intent] = float(confidence_scores(i))

            return {
                "intent": intent_pred,
                "confidence": float(max_confidence),
                "all_confidences": confidence_by_class,
                "text": text
            }

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_batch(self, texts):
        """
        Predict intents for multiple texts

        Args:
            texts: List of Turkish text inputs
        Returns:
            List of prediction dictionaries
        """
        if not texts:
            raise ValueError("Text list cannot be empty")

        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                results.append({
                    "intent": "UNKNOWN",
                    "confidence": 0.0,
                    "error": str(e),
                    "text": text
                })

        return results

    def save_model(self):
        """
        Save trained model to disk
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        try:
            model_data = {
                'classifier': self.classifier,
                'intent_labels':self.intent_labels,
                'is_trained:': self.is_trained
            }

            joblib.dump(model_data, self.model_path)

        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")

    def load_model(self):
        """
        Load trained model from the disk
        Returns: True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.model_path):
            return False

        try:
            model_data = joblib.load(self.model_path)

            self.classifier = model_data['classifier']
            self.intent_labels = model_data['intent_labels']
            self.is_trained = model_data['is_trained']

            return True

        except Exception as e:
            return False

    def get_supported_intents(self):
        """Get list of supported intent types"""
        return self.intent_labels.copy()

    def get_model_info(self):
        """Get model information"""
        return {
            "is_trained": self.is_trained,
            "supported_intents": self.intent_labels,
            "model_path": self.model_path,
            "berturk_loaded": self.berturk.is_loaded()
        }

def create_intent_classifier():
    """Convenience function to create intent classifier"""
    return IntentClassifier()

