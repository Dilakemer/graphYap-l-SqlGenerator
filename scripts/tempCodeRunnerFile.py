"""
Turkish NER Data Generator for SQL Query Understanding
Version: 2.0 - Fully Compatible with SQLGenerator
"""

import json
import random
from datetime import datetime, timedelta
import hashlib
from tqdm import tqdm
from collections import defaultdict

class TurkishNERGenerator:
    def __init__(self):
        # SQLGenerator schema'sıyla tam uyumlu yapı
        self.schema = {
            "TABLE_CUSTOMERS": {
                "display_columns": ["customer_id", "customer_name", "city", "registration_date"],
                "date_column": "registration_date",
                "sum_columns": ["total_purchases"],
                "avg_columns": ["total_purchases"],
                "countable_column": "customer_id"
            },
            "TABLE_ORDERS": {
                "display_columns": ["order_id", "customer_id", "order_date", "total_amount"],
                "date_column": "order_date",
                "sum_columns": ["total_amount"],
                "avg_columns": ["total_amount"],
                "countable_column": "order_id"
            },
            "TABLE_PRODUCTS": {
                "display_columns": ["product_id", "product_name", "category_id", "price"],
                "sum_columns": ["price"],
                "avg_columns": ["price"]
            },
            "TABLE_CATEGORIES": {
                "display_columns": ["category_id", "category_name"],
                "sum_columns": [],
                "avg_columns": []
            }
        }

        # SQLGenerator entity mapping ile tam uyumlu
        self.entity_map = {
            "tables": {
                "TABLE_CUSTOMERS": ["müşteri", "müşteriler", "client"],
                "TABLE_ORDERS": ["sipariş", "siparişler", "order"],
                "TABLE_PRODUCTS": ["ürün", "product", "mal"],
                "TABLE_CATEGORIES": ["kategori", "category"]
            },
            "columns": {
                "customer_id": ["müşteri no", "müşteri id"],
                "customer_name": ["müşteri adı", "isim"],
                "order_date": ["sipariş tarihi", "tarih"],
                "total_amount": ["toplam tutar", "miktar"],
                "price": ["fiyat", "ücret"],
                "category_name": ["kategori adı"]
            },
            "intents": {
                "SELECT": ["göster", "listele", "getir"],
                "COUNT": ["say", "adet", "kaç tane"],
                "SUM": ["toplam", "ne kadar"],
                "AVG": ["ortalama", "ort"],
                "MAX": ["en fazla", "maksimum"],
                "MIN": ["en az", "minimum"]
            },
            "time_filters": {
                "current_month": ["bu ay", "bu ay içinde"],
                "last_month": ["geçen ay", "önceki ay"],
                "current_year": ["bu yıl", "bu sene"],
                "last_year": ["geçen yıl", "önceki sene"],
                "last_7_days": ["son 7 gün", "geçen hafta"],
                "last_30_days": ["son 30 gün", "geçen ay"],
                "specific_date": ["{} tarihinde", "{} günü"]
            },
            "conditions": {
                "greater_than": [">", "fazla", "üzerinde"],
                "less_than": ["<", "az", "altında"],
                "equals": ["=", "eşit", "olan"]
            }
        }

        # JOIN ilişkileri
        self.relationships = [
            {
                "tables": ["TABLE_CUSTOMERS", "TABLE_ORDERS"],
                "join_key": "customer_id",
                "phrases": ["müşteri siparişleri", "müşterinin siparişleri"]
            },
            {
                "tables": ["TABLE_PRODUCTS", "TABLE_CATEGORIES"],
                "join_key": "category_id",
                "phrases": ["ürün kategorileri", "kategorilere göre ürünler"]
            }
        ]

        # Üretilen örneklerin hash'leri
        self.generated_hashes = set()

    def generate_dataset(self, size=100000):
        """Generate balanced dataset with all query types"""
        samples = []
        patterns = [
            ("simple_select", 0.25),
            ("select_with_columns", 0.2),
            ("time_filtered", 0.2),
            ("aggregation", 0.15),
            ("join_query", 0.1),
            ("conditional", 0.1)
        ]

        for pattern, ratio in patterns:
            count = int(size * ratio)
            for _ in tqdm(range(count), desc=f"Generating {pattern}"):
                sample = self._generate_sample(pattern)
                if sample and self._is_unique(sample):
                    samples.append(sample)

        random.shuffle(samples)
        return samples[:size]

    def _generate_sample(self, pattern_type):
        """Generate sample based on pattern type"""
        generator = getattr(self, f"_generate_{pattern_type}")
        return generator()

    def _is_unique(self, sample):
        """Check if sample is unique"""
        sample_hash = hashlib.md5(
            json.dumps(sample, sort_keys=True).encode()
        ).hexdigest()
        
        if sample_hash in self.generated_hashes:
            return False
            
        self.generated_hashes.add(sample_hash)
        return True

    # Pattern Generators ------------------------------------------------------

    def _generate_simple_select(self):
        """Basic SELECT * FROM table queries"""
        table, table_phrase = self._random_table()
        intent = random.choice(self.entity_map["intents"]["SELECT"])
        
        text = f"{intent} {table_phrase}"
        entities = [
            {"text": table_phrase, "label": f"TABLE_{table}", "start": len(intent)+1, "end": len(intent)+1+len(table_phrase)}
        ]
        
        return {"text": text, "entities": entities}

    def _generate_select_with_columns(self):
        """SELECT with specific columns"""
        table, table_phrase = self._random_table()
        columns = self._random_columns(table, max_cols=2)
        intent = random.choice(self.entity_map["intents"]["SELECT"])
        
        column_phrase = " ve ".join(c["text"] for c in columns)
        text = f"{intent} {column_phrase} {table_phrase}"
        
        entities = [
            {"text": table_phrase, "label": f"TABLE_{table}", 
             "start": text.find(table_phrase), "end": text.find(table_phrase)+len(table_phrase)}
        ]
        
        for col in columns:
            entities.append({
                "text": col["text"],
                "label": f"COLUMN_{col['name']}",
                "start": text.find(col["text"]),
                "end": text.find(col["text"])+len(col["text"])
            })
            
        return {"text": text, "entities": sorted(entities, key=lambda x: x["start"])}

    def _generate_time_filtered(self):
        """Queries with time filters"""
        table, table_phrase = self._random_table(with_date=True)
        time_filter = self._random_time_filter()
        intent = random.choice(self.entity_map["intents"]["SELECT"] + self.entity_map["intents"]["COUNT"])
        
        # Randomly arrange components
        if random.random() < 0.5:
            text = f"{time_filter['text']} {intent} {table_phrase}"
        else:
            text = f"{intent} {table_phrase} {time_filter['text']}"
        
        entities = [
            {"text": table_phrase, "label": f"TABLE_{table}", 
             "start": text.find(table_phrase), "end": text.find(table_phrase)+len(table_phrase)},
            {"text": time_filter["text"], "label": "TIME_FILTER",
             "start": text.find(time_filter["text"]), "end": text.find(time_filter["text"])+len(time_filter["text"])}
        ]
        
        if intent in self.entity_map["intents"]["COUNT"]:
            entities.append({
                "text": intent, "label": "INTENT_COUNT",
                "start": text.find(intent), "end": text.find(intent)+len(intent)
            })
            
        return {"text": text, "entities": sorted(entities, key=lambda x: x["start"])}

    def _generate_aggregation(self):
        """Aggregation queries (COUNT, SUM, AVG, MAX, MIN)"""
        agg_type = random.choice(["COUNT", "SUM", "AVG", "MAX", "MIN"])

        # Aggregation INTENT'i COUNT dışındaysa, uygun tabloları filtrele
        if agg_type == "COUNT":
            table, table_phrase = self._random_table()
        else:
            valid_tables = [
                t for t in self.schema
                if self.schema[t].get(f"{agg_type.lower()}_columns")
            ]
            if not valid_tables:
                return None  # Uygun tablo yoksa aggregation örneği üretme
            table = random.choice(valid_tables)
            table_phrase = random.choice(self.entity_map["tables"][table])

        if agg_type == "COUNT":
            text = f"{table_phrase} {random.choice(self.entity_map['intents']['COUNT'])}"
            entities = [
                {
                    "text": table_phrase,
                    "label": f"TABLE_{table}",
                    "start": 0,
                    "end": len(table_phrase)
                },
                {
                    "text": text.split()[-1],
                    "label": f"INTENT_{agg_type}",
                    "start": len(table_phrase) + 1,
                    "end": len(text)
                }
            ]
        else:
            # Güvenli column seçimi
            try:
                column = self._random_column(table, agg_type.lower())
            except ValueError:
                return None  # Uygun kolon yoksa örnek üretme

            column_start = len(table_phrase) + 1
            column_end = column_start + len(column["text"])
            intent_text = random.choice(self.entity_map['intents'][agg_type])
            intent_start = column_end + 1
            intent_end = intent_start + len(intent_text)

            text = f"{table_phrase} {column['text']} {intent_text}"
            entities = [
                {
                    "text": table_phrase,
                    "label": f"TABLE_{table}",
                    "start": 0,
                    "end": len(table_phrase)
                },
                {
                    "text": column["text"],
                    "label": f"COLUMN_{column['name']}",
                    "start": column_start,
                    "end": column_end
                },
                {
                    "text": intent_text,
                    "label": f"INTENT_{agg_type}",
                    "start": intent_start,
                    "end": intent_end
                }
            ]

        return {"text": text, "entities": entities}

    def _generate_join_query(self):
        """Multi-table JOIN queries"""
        relation = random.choice(self.relationships)
        table1, table2 = relation["tables"]
        table1_phrase = random.choice(self.entity_map["tables"][table1])
        table2_phrase = random.choice(self.entity_map["tables"][table2])
        
        if random.random() < 0.6:
            # Use natural join phrase
            join_phrase = random.choice(relation["phrases"])
            text = f"{random.choice(self.entity_map['intents']['SELECT'])} {join_phrase}"
            entities = [
                {"text": join_phrase, "label": "RELATIONSHIP",
                 "start": text.find(join_phrase), "end": text.find(join_phrase)+len(join_phrase)}
            ]
        else:
            # Explicit table join
            join_word = random.choice(["ile", "ve", "beraber"])
            text = f"{table1_phrase} {join_word} {table2_phrase}"
            entities = [
                {"text": table1_phrase, "label": f"TABLE_{table1}",
                 "start": 0, "end": len(table1_phrase)},
                {"text": table2_phrase, "label": f"TABLE_{table2}",
                 "start": len(table1_phrase)+1+len(join_word), 
                 "end": len(table1_phrase)+1+len(join_word)+len(table2_phrase)}
            ]
            
        return {"text": text, "entities": entities}

    def _generate_conditional(self):
        """Queries with WHERE conditions"""
        table, table_phrase = self._random_table()
        column = self._random_column(table)
        condition_type, condition_phrases = random.choice(list(self.entity_map["conditions"].items()))
        condition_phrase = random.choice(condition_phrases)
        value = self._generate_value_for_column(column["name"])
        
        text = f"{table_phrase} {column['text']} {condition_phrase} {value}"
        
        entities = [
            {"text": table_phrase, "label": f"TABLE_{table}",
             "start": 0, "end": len(table_phrase)},
            {"text": column["text"], "label": f"COLUMN_{column['name']}",
             "start": len(table_phrase)+1, "end": len(table_phrase)+1+len(column["text"])},
            {"text": str(value), "label": "VALUE",
             "start": len(table_phrase)+1+len(column["text"])+1+len(condition_phrase),
             "end": len(text)}
        ]
        
        return {"text": text, "entities": entities}

    # Helper Methods ----------------------------------------------------------

    def _random_table(self, with_date=False):
        """Select random table that meets criteria"""
        candidates = [
            t for t in self.schema 
            if not with_date or "date_column" in self.schema[t]
        ]
        table = random.choice(candidates)
        return table, random.choice(self.entity_map["tables"][table])

    def _random_column(self, table, col_type=None):
        """Get random column from table"""
        if col_type:
            candidates = self.schema[table].get(f"{col_type.lower()}_columns", [])
            if not candidates:
                raise ValueError(f"No {col_type.upper()} columns available in {table}")
            col_name = random.choice(candidates)
        else:
            col_name = random.choice(self.schema[table]["display_columns"])
            
        return {
            "name": col_name,
            "text": random.choice(self.entity_map["columns"].get(col_name, [col_name]))
        }


    def _random_columns(self, table, max_cols=2):
        """Get multiple random columns"""
        num = random.randint(1, max_cols)
        return [self._random_column(table) for _ in range(num)]

    def _random_time_filter(self):
        """Generate realistic time filter"""
        filter_type, phrases = random.choice(list(self.entity_map["time_filters"].items()))
        
        if "{}" in phrases[0]:  # Dynamic filter (last N days)
            n = random.choice([7, 30, 90])
            text = phrases[0].format(n)
            return {"type": filter_type, "text": text, "value": n}
        elif filter_type == "specific_date":
            date = self._random_date().strftime("%d.%m.%Y")
            text = phrases[0].format(date)
            return {"type": filter_type, "text": text, "value": date}
        else:
            text = random.choice(phrases)
            return {"type": filter_type, "text": text}

    def _random_date(self):
        """Generate random date within range"""
        start = datetime.now() - timedelta(days=365*2)
        end = datetime.now()
        return start + timedelta(days=random.randint(0, (end - start).days))

    def _generate_value_for_column(self, column_name):
        """Generate realistic values for conditions"""
        if "price" in column_name or "amount" in column_name:
            return round(random.uniform(10, 1000), 2)
        elif "date" in column_name:
            return self._random_date().strftime("%Y-%m-%d")
        elif "id" in column_name:
            return random.randint(1, 1000)
        else:
            return random.choice(["true", "false", "active", "inactive"])

    def save_dataset(self, filename="turkish_ner_dataset.json", size=100000):
        """Generate and save dataset to file"""
        dataset = self.generate_dataset(size)
        
        result = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "schema_version": "2.0",
                "compatible_with": "SQLGenerator v1.2+",
                "entity_types": [
                    "TABLE_CUSTOMERS", "TABLE_ORDERS", "TABLE_PRODUCTS", "TABLE_CATEGORIES",
                    "COLUMN_*", "INTENT_*", "TIME_FILTER", "VALUE", "RELATIONSHIP"
                ]
            },
            "samples": dataset
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Dataset saved to {filename} ({len(dataset)} samples)")

if __name__ == "__main__":
    generator = TurkishNERGenerator()
    generator.save_dataset(size=100000)