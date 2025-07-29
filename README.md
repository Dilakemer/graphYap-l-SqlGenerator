# 📊 graphYapı SQL Generator

Türkçe doğal dilde verilen kullanıcı sorgularını anlayarak ilişkisel veritabanı şeması üzerinden **doğru ve anlamlı SQL sorguları** otomatik olarak oluşturan bir sistemdir. Yapay zeka destekli bu proje, BERT tabanlı NER modeli ve grafik (graph) temelli JOIN çıkarım sistemi ile çalışmaktadır.

---

## 🚀 Proje Özeti

`graphYapı SQL Generator`, bir veritabanı şemasını **graph veri yapısı** ile modelliyor. Ardından, Türkçe cümlelerdeki yapıları analiz edip bu şema üzerinden SQL sorgusu oluşturuyor. Aşağıdaki bileşenleri bir arada kullanır:

- 🧠 BERT tabanlı Türkçe **NER modeli** (intent, tablo, kolon, tarih vb.)
- 🔗 JOIN yapıları için **graph tabanlı yol bulma**
- 🛠️ Akıllı **SQL üretim motoru**
- 🧪 **Sorgu doğrulama** (Validator)

---

## 🧠 Yapay Zeka ve NER Model Eğitimi

### 📌 Kullanılan Model

- Model: [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased)  
- Amaç: Türkçe cümlelerdeki **tablo, kolon, tarih, filtre, intent** gibi varlıkları belirlemek  
- Model türü: `TokenClassification` (NER)  
- Kütüphaneler: 🤗 Transformers, Datasets, PyTorch  

### 📚 Eğitim Verisi

- Format: JSON (etiketli cümleler)  
- 71 farklı etiket türü içerir: `TABLE_orders`, `COLUMN_total`, `DATE`, `INTENT_SUM`, vb.  
- 1500+ etiketlenmiş örnek  

#### Örnek Veri:
```json
{
  "text": "Bu ay en fazla sipariş veren müşteri kim?",
  "entities": [
    {"text": "bu ay", "label": "DATE"},
    {"text": "en fazla", "label": "AGGREGATION_MODIFIER"},
    {"text": "sipariş", "label": "TABLE_orders"},
    {"text": "müşteri", "label": "TABLE_customers"},
    {"text": "ad", "label": "COLUMN_customers.name"}
  ]
}
#### Eğitim Adımları:
python train_ner.py \
  --model_name dbmdz/bert-base-turkish-cased \
  --train_data ./data/train.json \
  --val_data ./data/val.json \
  --output_dir ./models/ner_model \
  --epochs 10 \
  --batch_size 16 \
  --fp16
✅ Eğitim sonrası model models/ner_model/ dizinine kaydedilir.
🔍 ####NER ile Sorgu Anlamlandırma
Kullanıcı:

"Bu ay en fazla sipariş veren müşteri kim?"

NER Modeli:

[
  {"text": "bu ay", "label": "DATE"},
  {"text": "en fazla", "label": "AGGREGATION_MODIFIER"},
  {"text": "sipariş", "label": "TABLE_orders"},
  {"text": "müşteri", "label": "TABLE_customers"},
  {"text": "ad", "label": "COLUMN_customers.name"}
🧩#### Veritabanı Şeması ve JOIN Çıkarımı
📁 schema_mapper.py
Tabloların kolon bilgilerini okur (data/schema.json)

Kolon → tablo eşleşmelerini yapar

📁 relation_mapper.py
Tablolar arası ilişkileri (data/relations.json) graph olarak modeller

find_join_path() ile otomatik JOIN sıralaması belirler

Örnek schema.json
json
Kopyala
Düzenle
{
  "orders": ["id", "customer_id", "order_date", "total"],
  "customers": ["id", "name"]
}
Örnek relations.json
json
Kopyala
Düzenle
[
  {"source_table": "orders", "source_column": "customer_id", "target_table": "customers", "target_column": "id"}
]
🛠️ SQL Generator
📁 sql_generator.py
NER çıktısı ve JOIN grafiği ile SQL oluşturur

Desteklediği yapılar:

Aggregation (SUM, COUNT, AVG)

GROUP BY, ORDER BY, LIMIT

Tarih filtreleri

Çoklu JOIN ve çoklu tablo desteği

Örnek SQL:
SELECT c.name
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.order_date BETWEEN '2025-07-01' AND '2025-07-31'
GROUP BY c.name
ORDER BY COUNT(*) DESC
LIMIT 1;
🧪 #### Sorgu Doğrulama (Validator)
📁 query_validator.py
Oluşturulan SQL’in eksik veya tutarsız kısımlarını kontrol eder:

📂 Proje Yapısı
graphYapı-SqlGenerator/
├── src/
│   ├── nlp/ner_model/              # BERT tabanlı Türkçe NER
│   ├── query_builder/
│   │   ├── sql_generator.py        # SQL üretim motoru
│   │   ├── schema_mapper.py        # Kolon-tablo eşlemesi
│   │   ├── relation_mapper.py      # JOIN çıkarımı
│   │   ├── query_templates.py      # SQL kalıpları
│   │   └── query_validator.py      # Sorgu doğrulama
│   └── utils/                      # Yardımcı fonksiyonlar
├── data/
│   ├── schema.json                 # Tablo şeması
│   └── relations.json              # Tablo ilişkileri
├── models/
│   └── ner_model/                  # Eğitilmiş NER modeli dosyaları
├── main.py                         # Uçtan uca demo
├── requirements.txt
└── README.md
⚙️ #### Kurulum:

git clone https://github.com/Dilakemer/graphYap-l-SqlGenerator.git
cd graphYap-l-SqlGenerator
pip install -r requirements.txt
⚠️ Eğitilmiş model dosyalarını models/ner_model/ altına eklemeyi unutmayın.

