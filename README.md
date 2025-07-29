cat << 'EOF' > README.md
# 📊 graphYapı SQL Generator


Türkçe doğal dilde verilen kullanıcı sorgularını anlayarak ilişkisel veritabanı şeması üzerinden **doğru ve anlamlı SQL sorguları** otomatik olarak oluşturan bir sistemdir. Yapay zeka destekli bu proje, BERT tabanlı NER modeli ve grafik (graph) temelli JOIN çıkarım sistemi ile çalışmaktadır.

---

## 🔍 Mimari


Yukarıdaki diyagramda:

1. **Kullanıcı** doğal dil sorgusunu gönderir.  
2. **NER Modülü** (BERT) sorgudan tablo, kolon, zaman, intent vb. çıkarır.  
3. **Schema & Relation Mapper** graph yapısıyla tablolar arasındaki JOIN yollarını belirler.  
4. **SQL Generator** en uygun SQL sorgusunu oluşturur.  
5. **Validator** sorgunun tutarlılığını ve eksiksizliğini kontrol eder.  

---

## 🚀 Proje Özeti

`graphYapı SQL Generator`, ilişkisel veritabanı şemasını **graph veri yapısı** ile modelleyip, Türkçe doğal dil sorgularını adım adım işleyerek:

- Entities çıkarımı (NER)  
- Tablo/kolon eşlemesi (schema mapping)  
- JOIN yolu keşfi (relation mapping)  
- SQL sentezi (generator)  
- Sorgu doğrulama (validator)  

bileşenlerini birlikte çalıştırır.

---

## 🧠 Yapay Zeka ve NER Model Eğitimi

### 📌 Kullanılan Model

- **Model**: [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)  
- **Görev**: Token Classification (NER)  
- **Kütüphaneler**: 🤗 Transformers, Datasets, PyTorch  

### 📚 Eğitim Verisi

- **Format**: JSON (etiketli cümleler)  
- **Örnek Sayısı**: 1500+ cümle  
- **Etiket Sayısı**: 71 farklı etiket (TABLE_*, COLUMN_*, DATE, INTENT_*, AGGREGATION_MODIFIER, VALUE, vb.)  

#### Örnek Kayıt
\`\`\`json
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
\`\`\`

### 🏋️‍♀️ Eğitim Prosedürü

Eğitim script’i `src/nlp/ner_model/train_ner.py` içinde:

\`\`\`bash
python src/nlp/ner_model/train_ner.py \
  --model_name dbmdz/bert-base-turkish-cased \
  --train_data ./data/train.json \
  --val_data ./data/val.json \
  --output_dir ./models/ner_model \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --fp16
\`\`\`

Eğitim sonunda `models/ner_model/` içine `pytorch_model.bin`, `config.json`, `tokenizer.json` vb. kaydedilir.

---

## 🔎 Doğrulama & Test

Test seti üzerinde:

- **Accuracy**: %92+  
- **F1-Score**: %90+  

Örnek:
> "Geçen ay en çok sipariş alan müşteri kim?"

Model çıktısı:
\`\`\`json
[
  {"text": "geçen ay", "label": "DATE"},
  {"text": "en çok", "label": "AGGREGATION_MODIFIER"},
  {"text": "sipariş", "label": "TABLE_orders"},
  {"text": "müşteri", "label": "TABLE_customers"}
]
\`\`\`

---

## 📁 Veritabanı Şeması ve JOIN Çıkarımı

### schema_mapper.py

- `data/schema.json` dosyasını okuyarak tablo ve kolon haritasını oluşturur.  
- `find_column_table(column_name)` fonksiyonu ile herhangi bir kolonun ait olduğu tabloyu döndürür.

### relation_mapper.py

- `data/relations.json` içindeki foreign key tanımlarını graph olarak yükler.  
- `find_join_path(start_table, end_table)` ile iki tablo arasındaki en kısa JOIN yolunu bulur.  

#### Örnek schema.json
\`\`\`json
{
  "orders": ["id", "customer_id", "order_date", "total"],
  "customers": ["id", "name", "region_id"],
  "regions": ["id", "region_name"]
}
\`\`\`

#### Örnek relations.json
\`\`\`json
[
  {"source_table": "orders", "source_column": "customer_id", "target_table": "customers", "target_column": "id"},
  {"source_table": "customers", "source_column": "region_id", "target_table": "regions", "target_column": "id"}
]
\`\`\`

---

## 🛠️ SQL Generator

### sql_generator.py

1. **Entities** listesini alır (NER çıktısı).  
2. `SchemaMapper` ile tabloları ve kolonları eşler.  
3. `RelationMapper` ile gerekli JOIN yolunu oluşturur.  
4. Filtre (`WHERE`), gruplama (`GROUP BY`), sıralama (`ORDER BY`), limit (`LIMIT`) ifadelerini ekler.  
5. Son SQL sorgusunu string olarak döner.

#### Örnek
\`\`\`sql
SELECT c.name, COUNT(o.id) AS order_count
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.order_date BETWEEN '2025-07-01' AND '2025-07-31'
GROUP BY c.name
ORDER BY order_count DESC
LIMIT 5;
\`\`\`

---

## ✔️ Sorgu Doğrulama

### query_validator.py

- Oluşturulan SQL'in:
  - Zorunlu filtreleri içerip içermediğini,
  - Aggregation varsa `GROUP BY` uyumunu,
  - Kolon-varlık tutarlılığını
  kontrol eder. Hata durumunda açıklayıcı exception fırlatır.

---

## 📂 Proje Yapısı

\`\`\`
graphYapı-SqlGenerator/
├── assets/                        
│   ├── logo.png                   
│   └── flowchart.png              
├── docs/
│   └── images/                    
│       └── architecture.png       
├── src/
│   ├── nlp/ner_model/             
│   │   ├── train_ner.py           
│   │   ├── inference.py          
│   │   └── utils.py               
│   ├── query_builder/             
│   │   ├── sql_generator.py       
│   │   ├── schema_mapper.py       
│   │   ├── relation_mapper.py     
│   │   ├── query_templates.py     
│   │   └── query_validator.py     
│   └── utils/                     
│       └── helpers.py             
├── data/                          
│   ├── train.json                 
│   ├── val.json                   
│   ├── test.json                  
│   ├── schema.json                
│   └── relations.json             
├── models/ner_model/              
│   ├── config.json                
│   ├── pytorch_model.bin          
│   └── tokenizer.json             
├── main.py                        
├── requirements.txt               
└── README.md                      
\`\`\`

---

## ⚙️ Kurulum & Çalıştırma

git clone https://github.com/Dilakemer/graphYap-l-SqlGenerator.git
cd graphYap-l-SqlGenerator
pip install -r requirements.txt
python main.py
