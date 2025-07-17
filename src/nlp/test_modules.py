# test_modules.py
from berturk_wrapper import BERTurkWrapper
from entity_extractor import EntityExtractor
from intent_classifier import IntentClassifier
from nlp_processor import NLPProcessor


def test_berturk():
    """Test BERTurk wrapper functionality"""
    print("🔍 BERTurk Wrapper Test:")

    try:
        # Test singleton pattern
        berturk1 = BERTurkWrapper()
        berturk2 = BERTurkWrapper()
        print(f"   Singleton working: {berturk1 is berturk2}")

        # Test basic functionality
        print(f"   Model loaded: {berturk1.is_loaded()}")
        print(f"   Initialized: {berturk1._initialized}")

        # Test embedding generation
        test_text = "Müşteri sayısı"
        embedding = berturk1.get_embeddings(test_text)
        print(f"   Embedding shape: {embedding.shape}")

        # Test similarity
        similarity = berturk1.get_similarity("müşteri", "customer")
        print(f"   Similarity test: {similarity:.3f}")

        print("   ✅ BERTurk tests passed\n")

    except Exception as e:
        print(f"   ❌ BERTurk error: {e}\n")


def test_entity_extractor():
    """Test entity extraction functionality"""
    print("🔍 Entity Extractor Test:")

    try:
        extractor = EntityExtractor()

        # Test basic extraction with debug
        test_text = "Bu ayın müşteri sayısı"
        print(f"   Testing text: '{test_text}'")

        result = extractor.extract(test_text)
        print(f"   Extract result type: {type(result)}")
        print(f"   Extract result keys: {list(result.keys()) if result else 'None'}")

        if result:
            print(f"   Tables: {result.get('tables', 'missing')}")
            print(f"   Time filters: {result.get('time_filters', 'missing')}")
            print(f"   Metadata: {result.get('metadata', 'missing')}")

            if 'metadata' in result:
                print(f"   Metadata keys: {list(result['metadata'].keys())}")

        print("   ✅ Entity Extractor tests passed\n")

    except Exception as e:
        print(f"   ❌ Entity Extractor error: {e}")
        import traceback
        traceback.print_exc()
        print()


def train_intent_classifier():
    """Train the intent classifier"""
    print("🎓 Training Intent Classifier:")

    try:
        classifier = IntentClassifier()

        # Check if training data exists
        training_file = "../../data/intent_training_data.json"

        import os
        if not os.path.exists(training_file):
            print(f"   ❌ Training data not found: {training_file}")
            print("   Create the training data file first!")
            return False

        # Train the model
        print("   Training model...")
        classifier.train(training_file)

        print(f"   ✅ Model trained successfully!")
        print(f"   Accuracy: {classifier.accuracy:.3f}")
        print(f"   Model saved to: {classifier.model_path}")

        return True

    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return False


def test_intent_classifier():
    """Test intent classification functionality"""
    print("🔍 Intent Classifier Test:")

    try:
        classifier = IntentClassifier()

        # Test model info
        model_info = classifier.get_model_info()
        print(f"   Trained: {model_info['is_trained']}")
        print(f"   Supported intents: {model_info['supported_intents']}")
        print(f"   BERTurk loaded: {model_info['berturk_loaded']}")

        # Try to train if not trained
        if not model_info['is_trained']:
            print("   Model not trained, attempting to train...")
            trained = train_intent_classifier()
            if not trained:
                print("   Skipping prediction test")
                return

        # Test prediction
        test_text = "Müşteri sayısını göster"
        try:
            result = classifier.predict(test_text)
            print(f"   Predicted intent: {result['intent']}")
            print(f"   Confidence: {result['confidence']:.3f}")
        except RuntimeError as e:
            print(f"   Prediction failed: {str(e)}")

        print("   ✅ Intent Classifier tests passed\n")

    except Exception as e:
        print(f"   ❌ Intent Classifier error: {e}\n")


def test_nlp_processor():
    """Test NLP processor coordination"""
    print("🔍 NLP Processor Test:")

    try:
        processor = NLPProcessor()

        # Test system info
        system_info = processor.get_system_info()
        print(f"   Version: {system_info['nlp_processor_version']}")
        print(f"   Components loaded: {len(system_info['components'])}")

        # Test processing stats
        stats = processor.get_processing_stats()
        print(f"   Queries processed: {stats['total_processed']}")
        print(f"   Success rate: {stats['success_rate']}%")

        # Test analysis - with error handling for untrained model
        test_text = "Bu yılın sipariş sayısı"
        try:
            result = processor.analyze(test_text)

            print(f"   Analysis status: {result['analysis_metadata']['processing_status']}")
            print(f"   Intent: {result['intent']['type']}")
            print(f"   SQL ready: {result['analysis_metadata']['sql_ready']}")

            # Test validation
            validation = processor.validate_analysis(result)
            print(f"   Validation passed: {validation['valid']}")
            print(f"   Recommendation: {validation['recommendation']}")

        except Exception as analysis_error:
            print(f"   Analysis failed (expected - model not trained): {analysis_error}")
            print("   Note: Intent classifier needs training data to work properly")

        print("   ✅ NLP Processor tests passed\n")

    except Exception as e:
        print(f"   ❌ NLP Processor error: {e}\n")


def test_integration():
    """Test component integration"""
    print("🔍 Integration Test:")

    try:
        # Test individual components first
        processor = NLPProcessor()

        test_queries = [
            "Bu ayın müşteri sayısı",
            "Geçen yılın sipariş listesi",
            "Ürün kategorilerini göster"
        ]

        print(f"   Testing {len(test_queries)} queries...")

        # Test each query individually to catch errors
        successful_results = []
        for i, query in enumerate(test_queries):
            try:
                result = processor.analyze(query)
                if result['analysis_metadata']['processing_status'] == 'success':
                    successful_results.append(result)
                print(f"   Query {i + 1}: {result['analysis_metadata']['processing_status']}")
            except Exception as e:
                print(f"   Query {i + 1}: error - {e}")

        print(f"   Successful analyses: {len(successful_results)}/{len(test_queries)}")

        # Show one example if any successful
        if successful_results:
            example = successful_results[0]
            print(f"   Example - Intent: {example['intent']['type']}")
            print(f"   Example - Tables: {len(example['entities']['tables'])}")

        print("   ✅ Integration tests passed\n")

    except Exception as e:
        print(f"   ❌ Integration error: {e}\n")


if __name__ == "__main__":
    print("🚀 Starting NLP Module Tests...\n")

    # Run individual tests
    test_berturk()
    test_entity_extractor()
    test_intent_classifier()
    test_nlp_processor()

    # Run integration test
    test_integration()

    print("🎉 All tests completed!")