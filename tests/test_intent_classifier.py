"""
Intent Classifier Tests
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.nlp.intent_classifier import IntentClassifier, create_intent_classifier


class TestIntentClassifier:
    """Test intent classifier functionality"""

    def test_classifier_initialization(self):
        """Test classifier initializes correctly"""
        classifier = IntentClassifier()

        assert classifier.intent_labels == ['SELECT', 'COUNT', 'SUM', 'AVG']
        assert classifier.is_trained == False
        assert classifier.accuracy == 0

    def test_create_intent_classifier(self):
        """Test convenience function"""
        classifier = create_intent_classifier()

        assert isinstance(classifier, IntentClassifier)

    def test_get_supported_intents(self):
        """Test supported intents method"""
        classifier = IntentClassifier()
        intents = classifier.get_supported_intents()

        expected_intents = ['SELECT', 'COUNT', 'SUM', 'AVG']
        assert intents == expected_intents

    def test_get_model_info(self):
        """Test model information"""
        classifier = IntentClassifier()
        info = classifier.get_model_info()

        assert info["is_trained"] == False
        assert info["supported_intents"] == ['SELECT', 'COUNT', 'SUM', 'AVG']
        assert "model_path" in info
        assert "berturk_loaded" in info

    def test_predict_without_training(self):
        """Test prediction without training raises error"""
        classifier = IntentClassifier()

        with pytest.raises(RuntimeError, match="Model not trained"):
            classifier.predict("müşteri sayısını hesapla")

    def test_predict_empty_text(self):
        """Test prediction with empty text"""
        classifier = IntentClassifier()

        with pytest.raises(ValueError, match="Text input cannot be empty"):
            classifier.predict("")

    def test_predict_batch_empty_list(self):
        """Test batch prediction with empty list"""
        classifier = IntentClassifier()

        with pytest.raises(ValueError, match="Text list cannot be empty"):
            classifier.predict_batch([])

    @patch('src.nlp.intent_classifier.joblib.load')
    @patch('src.nlp.intent_classifier.os.path.exists')
    def test_load_model_success(self, mock_exists, mock_load):
        """Test successful model loading"""
        # Mock file exists
        mock_exists.return_value = True

        # Mock loaded model data
        mock_load.return_value = {
            'classifier': MagicMock(),
            'intent_labels': ['SELECT', 'COUNT', 'SUM', 'AVG'],
            'is_trained': True
        }

        classifier = IntentClassifier()
        result = classifier.load_model()

        assert result == True
        assert classifier.is_trained == True
        mock_exists.assert_called_once()
        mock_load.assert_called_once()

    @patch('src.nlp.intent_classifier.os.path.exists')
    def test_load_model_file_not_exists(self, mock_exists):
        """Test model loading when file doesn't exist"""
        # Mock file doesn't exist
        mock_exists.return_value = False

        classifier = IntentClassifier()
        result = classifier.load_model()

        assert result == False
        mock_exists.assert_called_once()

    @patch('src.nlp.intent_classifier.joblib.load')
    @patch('src.nlp.intent_classifier.os.path.exists')
    def test_load_model_error(self, mock_exists, mock_load):
        """Test model loading error handling"""
        # Mock file exists but loading fails
        mock_exists.return_value = True
        mock_load.side_effect = Exception("Loading failed")

        classifier = IntentClassifier()
        result = classifier.load_model()

        assert result == False

    @patch('src.nlp.intent_classifier.joblib.dump')
    @patch('src.nlp.intent_classifier.os.makedirs')
    def test_save_model_success(self, mock_makedirs, mock_dump):
        """Test successful model saving"""
        classifier = IntentClassifier()
        classifier.is_trained = True

        classifier.save_model()

        mock_dump.assert_called_once()

    def test_save_model_untrained(self):
        """Test saving untrained model raises error"""
        classifier = IntentClassifier()

        with pytest.raises(RuntimeError, match="Cannot save untrained model"):
            classifier.save_model()

    @patch('src.nlp.intent_classifier.BERTurkWrapper')
    @patch('src.nlp.intent_classifier.json.load')
    @patch('src.nlp.intent_classifier.open')
    def test_prepare_training_data(self, mock_open, mock_json_load, mock_berturk):
        """Test training data preparation"""
        # Mock BERTurk
        mock_berturk_instance = MagicMock()
        mock_berturk.return_value = mock_berturk_instance
        mock_berturk_instance.get_embeddings_batch.return_value = np.random.rand(3, 768)

        # Mock training data
        mock_json_load.return_value = {
            'training_data': [
                {'text': 'müşteri listesi', 'intent': 'SELECT'},
                {'text': 'müşteri sayısı', 'intent': 'COUNT'},
                {'text': 'satış toplamı', 'intent': 'SUM'}
            ]
        }

        classifier = IntentClassifier()
        X, y = classifier.prepare_training_data('fake_path.json')

        assert X.shape == (3, 768)
        assert len(y) == 3
        assert list(y) == ['SELECT', 'COUNT', 'SUM']
        mock_berturk_instance.get_embeddings_batch.assert_called_once()

    @patch('src.nlp.intent_classifier.BERTurkWrapper')
    @patch('src.nlp.intent_classifier.LogisticRegression')
    @patch('src.nlp.intent_classifier.train_test_split')
    @patch('src.nlp.intent_classifier.accuracy_score')
    def test_train_success(self, mock_accuracy, mock_split, mock_lr, mock_berturk):
        """Test successful training"""
        # Mock BERTurk
        mock_berturk_instance = MagicMock()
        mock_berturk.return_value = mock_berturk_instance
        mock_berturk_instance.get_embeddings_batch.return_value = np.random.rand(4, 768)

        # Mock train_test_split
        mock_split.return_value = (
            np.random.rand(3, 768),  # X_train
            np.random.rand(1, 768),  # X_test
            np.array(['COUNT', 'SELECT', 'SUM']),  # y_train
            np.array(['AVG'])  # y_test
        )

        # Mock classifier
        mock_classifier = MagicMock()
        mock_lr.return_value = mock_classifier
        mock_classifier.predict.return_value = np.array(['AVG'])

        # Mock accuracy
        mock_accuracy.return_value = 0.85

        classifier = IntentClassifier()
        classifier.classifier = mock_classifier

        # Mock prepare_training_data
        with patch.object(classifier, 'prepare_training_data') as mock_prepare:
            mock_prepare.return_value = (np.random.rand(4, 768), np.array(['COUNT', 'SELECT', 'SUM', 'AVG']))

            # Mock save_model
            with patch.object(classifier, 'save_model') as mock_save:
                classifier.train('fake_training_file.json')

                assert classifier.is_trained == True
                assert classifier.accuracy == 0.85
                mock_classifier.fit.assert_called_once()
                mock_save.assert_called_once()

    @patch('src.nlp.intent_classifier.BERTurkWrapper')
    def test_predict_with_trained_model(self, mock_berturk):
        """Test prediction with trained model"""
        # Mock BERTurk
        mock_berturk_instance = MagicMock()
        mock_berturk.return_value = mock_berturk_instance
        mock_berturk_instance.get_embeddings.return_value = np.random.rand(768)

        # Mock trained classifier
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = np.array(['COUNT'])
        mock_classifier.predict_proba.return_value = np.array([[0.1, 0.8, 0.05, 0.05]])
        mock_classifier.classes_ = np.array(['SELECT', 'COUNT', 'SUM', 'AVG'])

        classifier = IntentClassifier()
        classifier.classifier = mock_classifier
        classifier.is_trained = True

        result = classifier.predict("müşteri sayısını hesapla")

        assert result["intent"] == "COUNT"
        assert result["confidence"] == 0.8
        assert "all_confidences" in result
        assert result["text"] == "müşteri sayısını hesapla"

    @patch('src.nlp.intent_classifier.BERTurkWrapper')
    def test_predict_batch_success(self, mock_berturk):
        """Test batch prediction"""
        # Mock BERTurk
        mock_berturk_instance = MagicMock()
        mock_berturk.return_value = mock_berturk_instance

        classifier = IntentClassifier()
        classifier.is_trained = True

        # Mock predict method
        with patch.object(classifier, 'predict') as mock_predict:
            mock_predict.side_effect = [
                {"intent": "COUNT", "confidence": 0.8},
                {"intent": "SELECT", "confidence": 0.9}
            ]

            texts = ["müşteri sayısı", "ürün listesi"]
            results = classifier.predict_batch(texts)

            assert len(results) == 2
            assert results[0]["intent"] == "COUNT"
            assert results[1]["intent"] == "SELECT"
            assert mock_predict.call_count == 2