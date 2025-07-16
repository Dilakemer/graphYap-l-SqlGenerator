# scripts/generate_intent_data.py
import json
import random


def generate_intent_training_data():
    """Optimized Turkish intent data generation"""

    # Base entities (singular only)
    entities = [
        "müşteri",
        "ürün",
        "sipariş",
        "çalışan",
        "kategori",
        "tedarikçi",
        "satış",
        "stok",
    ]

    # Intent patterns with specific verbs and formats
    patterns = {
        "SELECT": {
            "verbs": ["göster", "listele", "getir", "bul", "ver"],
            "formats": [
                "{entity} listesini {verb}",
                "{entity} bilgilerini {verb}",
                "{entity} kayıtlarını {verb}",
                "tüm {entity}leri {verb}",
                "{entity}ları {verb}",
            ],
        },
        "COUNT": {
            "verbs": ["hesapla", "bul", "söyle", "ver"],
            "formats": [
                "{entity} sayısını {verb}",
                "kaç {entity} var",
                "{entity} adedini {verb}",
                "toplam {entity} sayısı",
                "{entity} miktarını {verb}",
            ],
        },
        "SUM": {
            "verbs": ["hesapla", "bul", "söyle", "ver", "çıkar"],
            "formats": [
                "{entity} toplamını {verb}",
                "toplam {entity} miktarı",
                "{entity}lerin toplamı",
                "{entity} tutarını {verb}",
                "genel {entity} toplamı",
            ],
        },
        "AVG": {
            "verbs": ["hesapla", "bul", "söyle", "ver"],
            "formats": [
                "{entity} ortalamasını {verb}",
                "ortalama {entity} miktarı",
                "{entity}lerin ortalaması",
                "ortalama {entity} değeri",
            ],
        },
    }

    # Generate combinations
    training_data = []

    for intent, config in patterns.items():
        for entity in entities:
            for verb in config["verbs"]:
                for format_pattern in config["formats"]:
                    text = format_pattern.format(entity=entity, verb=verb)

                    # Quality filters
                    if (
                        len(text.split()) >= 2
                        and len(text) < 40
                        and text not in [item["text"] for item in training_data]
                    ):
                        training_data.append({"text": text, "intent": intent})

    # Add high-quality manual samples
    manual_samples = [
        # SELECT variations
        {"text": "müşteri listesi", "intent": "SELECT"},
        {"text": "ürün bilgileri", "intent": "SELECT"},
        {"text": "hangi müşteriler var", "intent": "SELECT"},
        {"text": "sipariş detayları", "intent": "SELECT"},
        # COUNT variations
        {"text": "kaç ürün satıldı", "intent": "COUNT"},
        {"text": "ne kadar sipariş", "intent": "COUNT"},
        {"text": "müşteri adet", "intent": "COUNT"},
        # SUM variations
        {"text": "toplam gelir", "intent": "SUM"},
        {"text": "satış tutarı", "intent": "SUM"},
        {"text": "genel toplam", "intent": "SUM"},
        # AVG variations
        {"text": "ortalama fiyat", "intent": "AVG"},
        {"text": "sipariş ortalaması", "intent": "AVG"},
    ]

    training_data.extend(manual_samples)

    # Remove duplicates and shuffle
    seen = set()
    unique_data = []
    for item in training_data:
        if item["text"] not in seen:
            seen.add(item["text"])
            unique_data.append(item)

    random.shuffle(unique_data)
    return unique_data


def save_training_data():
    """Generate and save optimized training data"""
    print("🤖 Generating optimized intent training data...")

    training_data = generate_intent_training_data()

    # Calculate statistics
    stats = {}
    for item in training_data:
        intent = item["intent"]
        stats[intent] = stats.get(intent, 0) + 1

    print("\n📊 Generated samples:")
    total = 0
    for intent, count in stats.items():
        print(f"  {intent}: {count} samples")
        total += count

    print(f"\n🎯 Total unique samples: {total}")

    # Save to JSON
    output = {
        "meta": {
            "total_samples": len(training_data),
            "intent_distribution": stats,
            "generation_method": "optimized_pattern_based",
            "quality_filters": "min_2_words, max_40_chars, no_duplicates",
        },
        "training_data": training_data,
    }

    with open("data/intent_training_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("✅ Optimized intent_training_data.json created!")

    # Show sample examples
    print("\n🔍 Sample examples:")
    for intent in stats.keys():
        examples = [item["text"] for item in training_data if item["intent"] == intent][
            :2
        ]
        print(f"\n{intent}:")
        for example in examples:
            print(f"  • {example}")


if __name__ == "__main__":
    save_training_data()
