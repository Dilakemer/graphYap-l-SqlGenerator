import json

with open("data/ner_training_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = {
    "meta": {
        "total_samples": len(data.get("samples", []))
    },
    "training_data": data["samples"]
}

with open("data/ner_training_data.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("✅ Format dönüştürüldü!")
