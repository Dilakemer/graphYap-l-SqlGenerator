# src/nlp/test_pipeline.py
"""
Simple Pipeline Test - Direct Import Approach
"""

import sys
from pathlib import Path

# Add specific file paths
sys.path.append(str(Path(__file__).parent))  # nlp directory
sys.path.append(str(Path(__file__).parent.parent / "query_builder"))  # query_builder directory

# Import directly from files
from nlp_processor import NLPProcessor
from sql_generator import SQLGenerator


def test_complete_pipeline():
    """Test NLP to SQL pipeline"""
    print("🚀 Testing NLP-to-SQL Pipeline")
    print("=" * 40)

    # Initialize
    print("📦 Initializing...")
    nlp = NLPProcessor()
    sql_gen = SQLGenerator()

    # Test queries
    queries = [
        "Bu ayın müşteri sayısı",
        "Geçen yıl sipariş toplamı"
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")

        try:
            # NLP Analysis
            nlp_result = nlp.analyze(query)
            print(f"✅ NLP: {nlp_result['intent']['type']} (conf: {nlp_result['intent']['confidence']:.3f})")

            # SQL Generation
            sql_result = sql_gen.generate_sql(nlp_result)

            if sql_result["success"]:
                print(f"✅ SQL: {sql_result['sql']}")
            else:
                print(f"❌ SQL Error: {sql_result['error']}")

        except Exception as e:
            print(f"💥 Error: {e}")


if __name__ == "__main__":
    test_complete_pipeline()