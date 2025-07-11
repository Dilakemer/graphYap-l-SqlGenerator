# src/nlp/berturk_test.py
from transformers import AutoTokenizer, AutoModel
import torch
import time


def test_berturk_loading():
    """BERTurk model yükleme ve temel test"""
    print("🤖 BERTurk model yükleniyor...")

    try:
        start_time = time.time()

        # Model ve tokenizer yükle
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")

        load_time = time.time() - start_time
        print(f"✅ Model yüklendi: {load_time:.2f} saniye")

        # Test prompt'ları
        test_prompts = [
            "müşteri listesini göster",
            "ürün sayısını hesapla",
            "bu ayın siparişlerini listele",
            "satış toplamını bul"
        ]

        print("\n🔍 Test prompt'ları:")
        for prompt in test_prompts:
            process_start = time.time()

            # Tokenization
            tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

            # Model inference
            with torch.no_grad():
                outputs = model(**tokens)

            process_time = (time.time() - process_start) * 1000

            print(f"  📝 '{prompt}'")
            print(f"     Tokens: {tokens['input_ids'].shape[1]} adet")
            print(f"     İşlem süresi: {process_time:.1f}ms")
            print(f"     Embedding shape: {outputs.last_hidden_state.shape}")
            print()

        print("🎉 BERTurk test başarılı!")
        return True

    except Exception as e:
        print(f"❌ BERTurk test başarısız: {e}")
        return False


def test_turkish_tokenization():
    """Türkçe tokenization testi"""
    print("🇹🇷 Türkçe tokenization testi...")

    try:
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

        # Türkçe karakterler ve ek'ler
        turkish_tests = [
            "müşteri",
            "müşteriler",
            "müşterilerin",
            "müşterilerimizin",
            "göster",
            "gösterir",
            "göstermek"
        ]

        for text in turkish_tests:
            tokens = tokenizer.tokenize(text)
            print(f"  '{text}' → {tokens}")

        print("✅ Türkçe tokenization çalışıyor!")
        return True

    except Exception as e:
        print(f"❌ Tokenization test başarısız: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("🚀 BERTurk Integration Test")
    print("=" * 50)

    # Model loading test
    loading_success = test_berturk_loading()

    print("\n" + "=" * 30)

    # Turkish tokenization test
    tokenization_success = test_turkish_tokenization()

    print("\n" + "=" * 30)
    print("📊 Test Özeti:")
    print(f"  Model Loading: {'✅' if loading_success else '❌'}")
    print(f"  Tokenization: {'✅' if tokenization_success else '❌'}")

    if loading_success and tokenization_success:
        print("\n🎉 Tüm testler başarılı! BERTurk hazır!")
    else:
        print("\n❌ Bazı testler başarısız!")