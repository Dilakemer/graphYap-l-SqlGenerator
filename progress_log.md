# Turkish NLP-SQL Project Progress

## WEEK 1 (13-19 Ocak 2025) - Foundation & Database ✅ COMPLETE
**Ana Hedef:** Database schema + Python connection + NLP foundation

### **Bu hafta tamamlanan:**
- **Database Foundation**
  - 8-table ERP schema design (customers, suppliers, categories, products, employees, orders, order_details, purchase_orders)
  - Foreign key relationships and performance indexes
  - PostgreSQL integration ve connection management
  - Turkish sample data generation system by using Faker library in Python (200+ realistic records)

- **Development Environment** 
  - PyCharm + Python 3.13 virtual environment setup
  - Git workflow setup (feature branches, clean commits)
  - Black formatter configuration
  - Organized project structure (src/, config/, data/, scripts/)

- **NLP Integration**
  - Successful BERTurk model integration 
  - Turkish tokenization ve morphological analysis working
  - Model loading test: 4.2s load time, ~80ms inference
  - Performance benchmarks established

- **Training Data System**
  - Systematic intent data generation script
  - 200+ Turkish prompt samples (SELECT, COUNT, SUM, AVG intents)
  - Pattern-based generation + manual quality samples
  - Duplicate filtering ve data quality assurance

### **Written scripts and modules:**
- `src/nlp/berturk_test.py` - BERTurk integration test
- `scripts/generate_sample_data.py` - Database sample data generator  
- `scripts/generate_intent_data.py` - Intent training data generator
- `data/create_database.sql` - Database schema
- Configuration files (.gitignore, pyproject.toml, requirements.txt)

### **Problems & Solutions:**
- Model download is taking some time → Cache mechanism will be implemented
- PostgreSQL VARCHAR limits → Column sizes optimized 
- Sample data FK constraints → Generation order fixed. At first I tried to insert products before categories. Thus,
  database couldn't understand the foreign key in the product table which category_id. Then I changed te order in the 
  insertion script.

---

## WEEK 2 (14 - 21.07.2025) - NLP Core Development
**Ana Hedef:** Intent classification + Entity extraction  
**Durum:** 🔄 Başlıyor

### **Bu hafta hedefleri:**
- [ ] Intent classification model implementation
- [ ] BERTurk + sklearn pipeline kurulumu
- [ ] Training ve test accuracy benchmarks
- [ ] Entity extraction (table/column detection)
- [ ] Basic SQL query generation prototype

### **Beklenen çıktı:** 
Working NLP pipeline: "müşteri sayısını hesapla" → Intent: COUNT, Entity: customers

---
