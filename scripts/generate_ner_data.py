import random
import json
import hashlib
from datetime import datetime, timedelta
from tqdm import tqdm
from src.query_builder.schema_mapper import SchemaMapper
class NERDataGenerator:
    def __init__(self, schema_mapper):
        self.schema_mapper = schema_mapper

        # Türkçe doğal ifadeler ve aliaslar
        self.table_aliases = {
            "customers": ["müşteri", "müşteriler", "firma", "şirket"],
            "products": ["ürün", "ürünler", "mal"],
            "orders": ["sipariş", "siparişler", "talep"],
            "categories": ["kategori", "kategoriler", "sınıf"],
            "suppliers": ["tedarikçi", "tedarikçiler", "sağlayıcı"],
            "employees": ["çalışan", "personel", "işçi"],
            "order_details": ["sipariş detay", "sipariş kalemi"],
            "purchase_orders": ["satın alma siparişi", "alım siparişi"],
            "employees": ["çalışan", "personel", "işçi", "maaş", "ücret", "gelir"],
        }

        self.intents = {
            "SELECT": ["göster", "listele", "getir", "ver"],
            "COUNT": ["kaç", "adet", "say"],
            "SUM": ["toplam", "tutar", "ne kadar", "toplam maaş", "maaş tutarı"],
            "AVG": ["ortalama", "vasati", "ort", "ortalama maaş"],
            "MAX": ["en yüksek", "maksimum", "en fazla"],
            "MIN": ["en düşük", "minimum", "en az"]
        }

        self.conditions = {
            "greater_than": ["büyük", "fazla", "üstünde", ">"],
            "less_than": ["küçük", "az", "altında", "<"],
            "equals": ["eşit", "olan", "="],
            "not_equals": ["değil", "olmayan", "!="]
        }

        self.time_filters = {
            "this_month": ["bu ay", "mevcut ay"],
            "last_month": ["geçen ay", "önceki ay"],
            "this_year": ["bu yıl", "mevcut yıl"],
            "last_year": ["geçen yıl", "önceki yıl"],
            "last_7_days": ["son 7 gün", "geçen hafta"],
            "last_30_days": ["son 30 gün", "geçen ay"],
            "today": ["bugün", "bu gün"]
        }

        # Üretilmiş hash seti (benzersiz kayıt için)
        self.generated_hashes = set()

    def generate_dataset(self, size=10000):
        samples = []
        pattern_weights = [
            ("simple_select", 0.2),
            ("select_with_columns", 0.15),
            ("time_filtered", 0.15),
            ("aggregation", 0.15),
            ("join_query", 0.15),
            ("conditional", 0.1),
            ("count_query", 0.1)
        ]
        total_weight = sum(w for _, w in pattern_weights)
        pattern_weights = [(p, w/total_weight) for p,w in pattern_weights]

        counts = {p: int(size * w) for p,w in pattern_weights}

        for pattern, count in counts.items():
            for _ in tqdm(range(count), desc=f"Generating {pattern}"):
                method = getattr(self, f"_generate_{pattern}")
                sample = method()
                if sample and self._is_unique(sample):
                    samples.append(sample)

        random.shuffle(samples)
        return samples[:size]

    def _is_unique(self, sample):
        h = hashlib.md5(json.dumps(sample, sort_keys=True).encode("utf-8")).hexdigest()
        if h in self.generated_hashes:
            return False
        self.generated_hashes.add(h)
        return True

    # --- Pattern generators ---

    def _generate_simple_select(self):
        table, table_name = self._random_table()
        table_alias = self._random_table_alias(table_name)

        intent = random.choice(self.intents["SELECT"])
        text = f"{intent} {table_alias} bilgilerini"

        entities = [
            self._entity(table_alias, f"TABLE_{table_name}", text)
        ]
        return {"text": text, "entities": entities}

    def _generate_select_with_columns(self):
        table, table_name = self._random_table()
        table_alias = self._random_table_alias(table_name)
        columns = self._random_columns(table_name, max_cols=2)

        intent = random.choice(self.intents["SELECT"])

        col_texts = [c["text"] for c in columns]
        col_phrase = " ve ".join(col_texts)
        text = f"{intent} {table_alias} için {col_phrase} bilgilerini"

        entities = [
            self._entity(table_alias, f"TABLE_{table_name}", text)
        ]
        for col in columns:
            entities.append(self._entity(col["text"], f"COLUMN_{col['name']}", text))

        return {"text": text, "entities": sorted(entities, key=lambda x: x["start"])}

    def _generate_time_filtered(self):
        table, table_name = self._random_table(with_date=True)
        table_alias = self._random_table_alias(table_name)
        time_filter_text = random.choice(sum(self.time_filters.values(), []))
        intent = random.choice(self.intents["SELECT"] + self.intents["COUNT"])

        # Pozisyonları ve cümle yapısını karıştır
        if random.random() < 0.5:
            text = f"{time_filter_text} {table_alias} {intent}"
        else:
            text = f"{intent} {table_alias} {time_filter_text}"

        entities = [
            self._entity(table_alias, f"TABLE_{table_name}", text),
            self._entity(time_filter_text, "TIME_FILTER", text)
        ]
        entities.append(self._entity(intent, f"INTENT_{intent.upper()}", text))

        return {"text": text, "entities": sorted(entities, key=lambda x: x["start"])}

    def _generate_aggregation(self):
        agg_type = random.choice(list(self.intents.keys() - {"SELECT"}))
        table, table_name = self._random_table()

        # Eğer sum/avg ise destekleniyor mu kontrolü
        if agg_type in ["SUM", "AVG"]:
            schema = self.schema_mapper.get_table_schema(table_name)
            cols = schema.get(f"{agg_type.lower()}_columns", [])
            if not cols:
                return None
            col_name = random.choice(cols)
            col_text = self._random_column_alias(col_name)
        else:
            # max, min, count için rastgele sütun veya tablo
            if agg_type == "COUNT":
                col_name = None
            else:
                # max,min için rastgele display_column seç
                schema = self.schema_mapper.get_table_schema(table_name)
                display_cols = schema.get("display_columns", [])
                if not display_cols:
                    return None
                col_name = random.choice(display_cols)
                col_text = self._random_column_alias(col_name)

        intent = random.choice(self.intents[agg_type])
        table_alias = self._random_table_alias(table_name)

        if agg_type == "COUNT":
            text = f"{table_alias} için {intent} sayısı"
            entities = [
                self._entity(table_alias, f"TABLE_{table_name}", text),
                self._entity(intent, f"INTENT_{agg_type}", text)
            ]
        else:
            text = f"{table_alias} tablosundaki {col_text} sütununun {intent} değeri"
            entities = [
                self._entity(table_alias, f"TABLE_{table_name}", text),
                self._entity(col_text, f"COLUMN_{col_name}", text),
                self._entity(intent, f"INTENT_{agg_type}", text)
            ]

        return {"text": text, "entities": sorted(entities, key=lambda x: x["start"])}

    def _generate_join_query(self):
        # Basitçe, iki tablo seç, "ile" bağlacı kullan
        tables = self.schema_mapper.get_all_tables()
        t1, t2 = random.sample(tables, 2)
        t1_alias = self._random_table_alias(t1)
        t2_alias = self._random_table_alias(t2)

        join_words = ["ile", "ve", "birlikte"]
        join_word = random.choice(join_words)

        intent = random.choice(self.intents["SELECT"])

        text = f"{t1_alias} {join_word} {t2_alias} bilgilerini {intent}"

        entities = [
            self._entity(t1_alias, f"TABLE_{t1}", text),
            self._entity(t2_alias, f"TABLE_{t2}", text),
            self._entity(intent, f"INTENT_{intent.upper()}", text)
        ]
        return {"text": text, "entities": sorted(entities, key=lambda x: x["start"])}

    def _generate_conditional(self):
        table, table_name = self._random_table()
        table_alias = self._random_table_alias(table_name)
        schema = self.schema_mapper.get_table_schema(table_name)
        col_name = None

        # Random uygun column seç
        for col in schema.get("display_columns", []):
            # Eğer numeric veya date gibi koşul için uygun (örnek basitçe string hariç)
            if "date" in col or "price" in col or "amount" in col or "quantity" in col or "stock" in col or "id" in col:
                col_name = col
                break
        if not col_name:
            col_name = random.choice(schema.get("display_columns", []))

        col_text = self._random_column_alias(col_name)
        cond_type = random.choice(list(self.conditions.keys()))
        cond_word = random.choice(self.conditions[cond_type])

        # Değer üret (sütun tipine göre basit)
        val = self._generate_value_for_column(col_name)

        intent = random.choice(self.intents["SELECT"])

        text = f"{table_alias} tablosunda {col_text} {cond_word} {val} olanlar {intent}"

        entities = [
            self._entity(table_alias, f"TABLE_{table_name}", text),
            self._entity(col_text, f"COLUMN_{col_name}", text),
            self._entity(cond_word, "CONDITION", text),
            self._entity(str(val), "VALUE", text),
            self._entity(intent, f"INTENT_{intent.upper()}", text)
        ]
        return {"text": text, "entities": sorted(entities, key=lambda x: x["start"])}

    def _generate_count_query(self):
        table, table_name = self._random_table()
        table_alias = self._random_table_alias(table_name)
        intent = random.choice(self.intents["COUNT"])

        text = f"{table_alias} sayısı {intent}"

        entities = [
            self._entity(table_alias, f"TABLE_{table_name}", text),
            self._entity(intent, f"INTENT_COUNT", text)
        ]
        return {"text": text, "entities": entities}

    # --- Yardımcı metodlar ---

    def _random_table(self, with_date=False):
        tables = self.schema_mapper.get_all_tables()
        if with_date:
            tables = [t for t in tables if self.schema_mapper.get_table_schema(t).get("date_column")]
            if not tables:
                tables = self.schema_mapper.get_all_tables()
        table = random.choice(tables)
        return table, table

    def _random_table_alias(self, table_name):
        return random.choice(self.table_aliases.get(table_name, [table_name]))

    def _random_column_alias(self, column_name):
        # Basitçe kolon adı veya alias
        if column_name == "salary":
            return random.choice(["maaş", "ücret", "gelir"])
        return column_name.replace("_", " ")

    def _random_columns(self, table_name, max_cols=2):
        schema = self.schema_mapper.get_table_schema(table_name)
        cols = schema.get("display_columns", [])
        num = random.randint(1, min(max_cols, len(cols)))
        selected = random.sample(cols, num)
        return [{"name": c, "text": self._random_column_alias(c)} for c in selected]

    def _generate_value_for_column(self, col_name):
        if "price" in col_name or "amount" in col_name:
            return round(random.uniform(10, 1000), 2)
        if "date" in col_name:
            # Tarih biçimi
            d = datetime.now() - timedelta(days=random.randint(0, 365))
            return d.strftime("%Y-%m-%d")
        if "quantity" in col_name or "stock" in col_name or "id" in col_name:
            return random.randint(1, 1000)
        # Default string değer
        return random.choice(["aktif", "pasif", "beklemede"])

    def _entity(self, text, label, full_text):
        """Entity dict with correct start/end in full_text"""
        start = full_text.find(text)
        if start == -1:
            # Eğer bulunamazsa 0 konabilir ya da None
            start = 0
        end = start + len(text)
        return {"text": text, "label": label, "start": start, "end": end}

    def save_dataset(self, filename="ner_training_data.json", size=10000):
        data = self.generate_dataset(size)
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "schema_tables": self.schema_mapper.get_all_tables(),
            "sample_size": size
        }
        result = {
            "metadata": metadata,
            "samples": data
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Dataset saved: {filename} ({len(data)} samples)")

if __name__ == "__main__":
    sm = SchemaMapper()
    generator = NERDataGenerator(sm)
    generator.save_dataset(size=50000)
