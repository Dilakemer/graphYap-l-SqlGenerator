# scripts/generate_ner_data.py
"""
Turkish NER Training Data Generator for NLP-SQL Project
Generates BIO-tagged entity data with comprehensive time support
"""

import json
import random
from datetime import datetime, timedelta
from itertools import product


class NERDataGenerator:
    """
    Generates comprehensive NER training data for Turkish NLP-SQL system
    Supports: Tables, Time filters (basic + advanced), Intents, Actions
    """

    def __init__(self):
        # Table entities
        self.table_entities = {
            "TABLE_CUSTOMERS": ["müşteri", "müşteriler", "client", "firma", "şirket"],
            "TABLE_PRODUCTS": ["ürün", "ürünler", "product", "mal", "eşya", "stok"],
            "TABLE_ORDERS": ["sipariş", "siparişler", "order", "satış", "satışlar"],
            "TABLE_CATEGORIES": ["kategori", "kategoriler", "grup", "gruplar", "tür", "sınıf"],
            "TABLE_SUPPLIERS": ["tedarikçi", "tedarikçiler", "supplier", "sağlayıcı"],
            "TABLE_EMPLOYEES": ["çalışan", "çalışanlar", "personel", "employee", "memur"],
            "TABLE_ORDER_DETAILS": ["sipariş detay", "sipariş detayı", "sipariş detayları", "order details"],
            "TABLE_PURCHASE_ORDERS": ["satın alma", "alım", "satın alma siparişi", "alım siparişi"]
        }

        # Intent/Action entities
        self.intent_entities = {
            "INTENT_SELECT": ["göster", "listele", "getir", "bul", "ver", "çıkar", "göstersin"],
            "INTENT_COUNT": ["sayı", "sayısı", "sayın", "adet", "adedi", "kaç", "tane", "miktar"],
            "INTENT_SUM": ["toplam", "toplamı", "sum", "total", "tutar", "tutarı"],
            "INTENT_AVG": ["ortalama", "ortalaması", "average", "avg", "mean"]
        }

        # Basic time entities
        self.basic_time_entities = {
            "TIME_CURRENT_MONTH": ["bu ay", "bu ayın", "bu ayki", "mevcut ay", "şimdiki ay"],
            "TIME_CURRENT_YEAR": ["bu yıl", "bu yılın", "bu yıla ait", "mevcut yıl", "şimdiki yıl"],
            "TIME_LAST_MONTH": ["geçen ay", "geçen ayın", "önceki ay", "önceki ayın"],
            "TIME_LAST_YEAR": ["geçen yıl", "geçen yılın", "önceki yıl", "önceki yılın"],
            "TIME_TODAY": ["bugün", "bugünkü", "bu gün", "bu günkü"],
            "TIME_CURRENT_WEEK": ["bu hafta", "bu haftanın", "bu haftaki", "mevcut hafta"],
            "TIME_LAST_WEEK": ["geçen hafta", "geçen haftanın", "önceki hafta", "önceki haftanın"]
        }

        # Advanced time entities - Relative periods
        self.relative_time_entities = {
            "TIME_LAST_N_DAYS": ["son {} gün", "geçen {} gün", "önceki {} gün"],
            "TIME_LAST_N_WEEKS": ["son {} hafta", "geçen {} hafta", "önceki {} hafta"],
            "TIME_LAST_N_MONTHS": ["son {} ay", "geçen {} ay", "önceki {} ay"],
            "TIME_LAST_N_YEARS": ["son {} yıl", "geçen {} yıl", "önceki {} yıl"]
        }

        # Quarter entities
        self.quarter_entities = {
            "TIME_Q1": ["1. çeyrek", "birinci çeyrek", "ilk çeyrek", "Q1"],
            "TIME_Q2": ["2. çeyrek", "ikinci çeyrek", "Q2"],
            "TIME_Q3": ["3. çeyrek", "üçüncü çeyrek", "Q3"],
            "TIME_Q4": ["4. çeyrek", "dördüncü çeyrek", "son çeyrek", "Q4"],
            "TIME_CURRENT_QUARTER": ["bu çeyrek", "mevcut çeyrek", "şimdiki çeyrek"],
            "TIME_LAST_QUARTER": ["geçen çeyrek", "önceki çeyrek"]
        }

        # Specific date patterns
        self.date_patterns = [
            "dd.mm.yyyy",  # 21.07.2022
            "dd/mm/yyyy",  # 21/07/2022
            "yyyy-mm-dd",  # 2022-07-21
            "dd mm yyyy",  # 21 07 2022
            "mm/yyyy",  # 07/2022
            "yyyy"  # 2022
        ]

        # Action verbs
        self.action_verbs = ["göster", "listele", "getir", "bul", "hesapla", "çıkar", "ver"]

        # Common numbers for relative dates
        self.relative_numbers = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90]

    def generate_comprehensive_ner_data(self, total_samples=2000):
        """Generate comprehensive NER training data"""
        print("🚀 Generating comprehensive Turkish NER training data...")

        training_data = []

        # Generate basic patterns
        training_data.extend(self._generate_basic_patterns(400))

        # Generate relative time patterns
        training_data.extend(self._generate_relative_time_patterns(400))

        # Generate quarter patterns
        training_data.extend(self._generate_quarter_patterns(200))

        # Generate specific date patterns
        training_data.extend(self._generate_specific_date_patterns(300))

        # Generate complex multi-entity patterns
        training_data.extend(self._generate_complex_patterns(400))

        # Generate edge cases
        training_data.extend(self._generate_edge_cases(300))

        # Remove duplicates and shuffle
        unique_data = self._remove_duplicates(training_data)
        random.shuffle(unique_data)

        return unique_data[:total_samples]

    def _generate_basic_patterns(self, count):
        """Generate basic table + intent + basic time patterns"""
        patterns = []

        for _ in range(count):
            # Random selections
            table_entity, table_words = random.choice(list(self.table_entities.items()))
            table_word = random.choice(table_words)

            intent_entity, intent_words = random.choice(list(self.intent_entities.items()))
            intent_word = random.choice(intent_words)

            action_verb = random.choice(self.action_verbs)

            # Sometimes add time
            if random.random() < 0.6:  # 60% chance
                time_entity, time_words = random.choice(list(self.basic_time_entities.items()))
                time_word = random.choice(time_words)

                # Different sentence structures
                templates = [
                    f"{time_word} {table_word} {intent_word}",
                    f"{table_word} {intent_word} {time_word}",
                    f"{time_word} {table_word} {intent_word} {action_verb}",
                    f"{table_word} {intent_word} {time_word} {action_verb}"
                ]

                text = random.choice(templates)
                entities = self._extract_entities_from_text(text, {
                    table_word: table_entity,
                    intent_word: intent_entity,
                    time_word: time_entity,
                    action_verb: "ACTION_VERB"
                })
            else:
                # No time entity
                templates = [
                    f"{table_word} {intent_word}",
                    f"{table_word} {intent_word} {action_verb}",
                    f"{table_word} {action_verb}"
                ]

                text = random.choice(templates)
                entities = self._extract_entities_from_text(text, {
                    table_word: table_entity,
                    intent_word: intent_entity,
                    action_verb: "ACTION_VERB"
                })

            patterns.append({
                "text": text,
                "entities": entities
            })

        return patterns

    def _generate_relative_time_patterns(self, count):
        """Generate relative time patterns: son 3 ay, son 10 gün etc."""
        patterns = []

        for _ in range(count):
            # Random selections
            table_entity, table_words = random.choice(list(self.table_entities.items()))
            table_word = random.choice(table_words)

            intent_entity, intent_words = random.choice(list(self.intent_entities.items()))
            intent_word = random.choice(intent_words)

            # Generate relative time
            time_type = random.choice(list(self.relative_time_entities.keys()))
            time_template = random.choice(self.relative_time_entities[time_type])
            number = random.choice(self.relative_numbers)
            relative_time = time_template.format(number)

            # Different sentence structures
            templates = [
                f"{relative_time} {table_word} {intent_word}",
                f"{table_word} {intent_word} {relative_time}",
                f"{relative_time} {table_word} {intent_word} göster",
                f"{relative_time}daki {table_word} {intent_word}"
            ]

            text = random.choice(templates)
            entities = self._extract_entities_from_text(text, {
                str(number): "TIME_NUMBER",
                relative_time.split()[-1]: "TIME_UNIT",  # gün, hafta, ay, yıl
                table_word: table_entity,
                intent_word: intent_entity
            })

            patterns.append({
                "text": text,
                "entities": entities
            })

        return patterns

    def _generate_quarter_patterns(self, count):
        """Generate quarter-based patterns"""
        patterns = []
        years = [2020, 2021, 2022, 2023, 2024]

        for _ in range(count):
            table_entity, table_words = random.choice(list(self.table_entities.items()))
            table_word = random.choice(table_words)

            intent_entity, intent_words = random.choice(list(self.intent_entities.items()))
            intent_word = random.choice(intent_words)

            # Quarter patterns
            if random.random() < 0.5:
                # Basic quarter
                quarter_entity, quarter_words = random.choice(list(self.quarter_entities.items()))
                quarter_word = random.choice(quarter_words)

                templates = [
                    f"{quarter_word} {table_word} {intent_word}",
                    f"{table_word} {intent_word} {quarter_word}",
                    f"{quarter_word}deki {table_word} {intent_word}"
                ]
            else:
                # Year + quarter
                year = random.choice(years)
                quarter_entity, quarter_words = random.choice(list(self.quarter_entities.items()))
                quarter_word = random.choice(quarter_words)

                templates = [
                    f"{year} {quarter_word} {table_word} {intent_word}",
                    f"{year} yılın {quarter_word} {table_word} {intent_word}",
                    f"{quarter_word} {year} {table_word} {intent_word}"
                ]

            text = random.choice(templates)
            entities = self._extract_entities_from_text(text, {
                table_word: table_entity,
                intent_word: intent_entity,
                quarter_word: quarter_entity
            })

            patterns.append({
                "text": text,
                "entities": entities
            })

        return patterns

    def _generate_specific_date_patterns(self, count):
        """Generate specific date patterns: 21.07.2022, 2022-07-21 etc."""
        patterns = []

        for _ in range(count):
            table_entity, table_words = random.choice(list(self.table_entities.items()))
            table_word = random.choice(table_words)

            intent_entity, intent_words = random.choice(list(self.intent_entities.items()))
            intent_word = random.choice(intent_words)

            # Generate random date
            year = random.randint(2020, 2024)
            month = random.randint(1, 12)
            day = random.randint(1, 28)  # Safe day range

            # Different date formats
            date_formats = [
                f"{day:02d}.{month:02d}.{year}",  # 21.07.2022
                f"{day:02d}/{month:02d}/{year}",  # 21/07/2022
                f"{year}-{month:02d}-{day:02d}",  # 2022-07-21
                f"{day} {month} {year}",  # 21 7 2022
                f"{month:02d}/{year}",  # 07/2022
                str(year)  # 2022
            ]

            date_str = random.choice(date_formats)

            # Templates
            templates = [
                f"{date_str} {table_word} {intent_word}",
                f"{table_word} {intent_word} {date_str}",
                f"{date_str} tarihli {table_word} {intent_word}",
                f"{date_str} tarihi {table_word} {intent_word}",
                f"{table_word} {intent_word} {date_str} tarih"
            ]

            text = random.choice(templates)
            entities = self._extract_entities_from_text(text, {
                date_str: "TIME_SPECIFIC_DATE",
                table_word: table_entity,
                intent_word: intent_entity
            })

            patterns.append({
                "text": text,
                "entities": entities
            })

        return patterns

    def _generate_complex_patterns(self, count):
        """Generate complex multi-entity patterns"""
        patterns = []

        for _ in range(count):
            # Multiple tables
            if random.random() < 0.3:
                table1_entity, table1_words = random.choice(list(self.table_entities.items()))
                table1_word = random.choice(table1_words)

                table2_entity, table2_words = random.choice(list(self.table_entities.items()))
                table2_word = random.choice(table2_words)

                intent_entity, intent_words = random.choice(list(self.intent_entities.items()))
                intent_word = random.choice(intent_words)

                templates = [
                    f"{table1_word} ve {table2_word} {intent_word}",
                    f"{table1_word} {table2_word} {intent_word}",
                    f"{table1_word} ile {table2_word} {intent_word}"
                ]

                text = random.choice(templates)
                entities = self._extract_entities_from_text(text, {
                    table1_word: table1_entity,
                    table2_word: table2_entity,
                    intent_word: intent_entity
                })

            # Time range patterns
            else:
                table_entity, table_words = random.choice(list(self.table_entities.items()))
                table_word = random.choice(table_words)

                intent_entity, intent_words = random.choice(list(self.intent_entities.items()))
                intent_word = random.choice(intent_words)

                # Date ranges
                year1 = random.randint(2020, 2023)
                year2 = year1 + random.randint(1, 2)

                templates = [
                    f"{year1} ile {year2} arası {table_word} {intent_word}",
                    f"{year1}-{year2} {table_word} {intent_word}",
                    f"{year1} den {year2} ye kadar {table_word} {intent_word}"
                ]

                text = random.choice(templates)
                entities = self._extract_entities_from_text(text, {
                    str(year1): "TIME_RANGE_START",
                    str(year2): "TIME_RANGE_END",
                    table_word: table_entity,
                    intent_word: intent_entity
                })

            patterns.append({
                "text": text,
                "entities": entities
            })

        return patterns

    def _generate_edge_cases(self, count):
        """Generate edge cases and challenging patterns"""
        patterns = []

        edge_templates = [
            # Typos and variations
            "müşetri sayısı",  # Typo
            "ürünlerin toplamı",  # Different form
            "siparişlere ait veriler",  # Complex structure
            "kategorilere göre grupla",  # Grouping
            "en çok satan ürünler",  # Superlative
            "hiç sipariş vermeyen müşteriler",  # Negation
        ]

        for template in edge_templates[:count]:
            # Simple entity extraction for edge cases
            entities = []
            words = template.split()

            for i, word in enumerate(words):
                # Basic pattern matching for edge cases
                if "müş" in word:
                    entities.append({
                        "start": template.find(word),
                        "end": template.find(word) + len(word),
                        "label": "TABLE_CUSTOMERS",
                        "text": word
                    })
                elif "ürün" in word:
                    entities.append({
                        "start": template.find(word),
                        "end": template.find(word) + len(word),
                        "label": "TABLE_PRODUCTS",
                        "text": word
                    })
                elif "sipariş" in word:
                    entities.append({
                        "start": template.find(word),
                        "end": template.find(word) + len(word),
                        "label": "TABLE_ORDERS",
                        "text": word
                    })
                elif word in ["sayı", "toplam", "veri"]:
                    entities.append({
                        "start": template.find(word),
                        "end": template.find(word) + len(word),
                        "label": "INTENT_COUNT",
                        "text": word
                    })

            patterns.append({
                "text": template,
                "entities": entities
            })

        return patterns

    def _extract_entities_from_text(self, text, entity_map):
        """Extract entities with BIO tagging from text"""
        entities = []

        for word, label in entity_map.items():
            if word in text:
                start_idx = text.find(word)
                end_idx = start_idx + len(word)

                entities.append({
                    "start": start_idx,
                    "end": end_idx,
                    "label": label,
                    "text": word
                })

        # Sort by start position
        entities.sort(key=lambda x: x["start"])
        return entities

    def _remove_duplicates(self, data):
        """Remove duplicate training samples"""
        seen_texts = set()
        unique_data = []

        for item in data:
            if item["text"] not in seen_texts:
                seen_texts.add(item["text"])
                unique_data.append(item)

        return unique_data

    def get_statistics(self, data):
        """Get dataset statistics"""
        stats = {
            "total_samples": len(data),
            "entity_distribution": {},
            "average_entities_per_sample": 0,
            "unique_entity_types": set()
        }

        total_entities = 0

        for sample in data:
            entities = sample.get("entities", [])
            total_entities += len(entities)

            for entity in entities:
                label = entity["label"]
                stats["entity_distribution"][label] = stats["entity_distribution"].get(label, 0) + 1
                stats["unique_entity_types"].add(label)

        stats["average_entities_per_sample"] = round(total_entities / len(data), 2) if data else 0
        stats["unique_entity_types"] = list(stats["unique_entity_types"])

        return stats

    def save_ner_data(self, output_file="../data/ner_training_data.json"):
        """Generate and save NER training data"""
        print("🤖 Generating comprehensive Turkish NER training data...")

        # Generate data
        training_data = self.generate_comprehensive_ner_data(2000)

        # Get statistics
        stats = self.get_statistics(training_data)

        # Prepare output
        output = {
            "meta": {
                "total_samples": stats["total_samples"],
                "entity_distribution": stats["entity_distribution"],
                "average_entities_per_sample": stats["average_entities_per_sample"],
                "unique_entity_types": stats["unique_entity_types"],
                "generation_method": "comprehensive_pattern_based_with_advanced_time_support",
                "supported_features": [
                    "basic_time_entities",
                    "relative_time_periods",
                    "quarter_patterns",
                    "specific_dates",
                    "complex_multi_entity",
                    "edge_cases"
                ]
            },
            "training_data": training_data
        }

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n📊 NER Training Data Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Unique entity types: {len(stats['unique_entity_types'])}")
        print(f"  Average entities per sample: {stats['average_entities_per_sample']}")

        print(f"\n🔍 Entity Distribution:")
        for entity_type, count in sorted(stats['entity_distribution'].items()):
            print(f"  {entity_type}: {count}")

        print(f"\n✅ NER training data saved to: {output_file}")

        # Show sample examples
        print(f"\n🔍 Sample Examples:")
        for i, sample in enumerate(random.sample(training_data, min(5, len(training_data)))):
            print(f"\n  Example {i + 1}:")
            print(f"    Text: '{sample['text']}'")
            print(f"    Entities: {len(sample['entities'])}")
            for entity in sample['entities']:
                print(f"      - '{entity['text']}' → {entity['label']}")


if __name__ == "__main__":
    generator = NERDataGenerator()
    generator.save_ner_data()