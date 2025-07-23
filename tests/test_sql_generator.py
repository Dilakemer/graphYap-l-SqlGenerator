from src.query_builder.sql_generator import SQLGenerator
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_sql_generator_with_joins():
    generator = SQLGenerator()
    nlp_analysis = {
        "intent": {"type": "SELECT", "confidence": 0.95},
        "entities": {
            "tables": [{"table": "orders"}, {"table": "customers"}],
            "time_filters": []
        },
        "analysis_metadata": {"sql_ready": True}
    }
    result = generator.generate_sql(nlp_analysis)
    print(result)
    assert result["success"] is True
    assert "JOIN" in result["sql"]  # Join sorgusu var mı kontrol et

if __name__ == "__main__":
    test_sql_generator_with_joins()


