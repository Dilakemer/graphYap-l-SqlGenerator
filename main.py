# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from src.nlp.entity_extractor import create_entity_extractor
from src.query_builder.sql_generator import create_sql_generator

app = FastAPI()

extractor = create_entity_extractor()
sql_generator = create_sql_generator()

class QueryRequest(BaseModel):
    text: str

class SQLResponse(BaseModel):
    success: bool
    sql: Optional[str]
    error: Optional[str]
    intent: Optional[str]
    table: Optional[str]
    metadata: Optional[Dict[str, Any]]

@app.post("/ask", response_model=SQLResponse)
def ask(request: QueryRequest):
    if not extractor.is_ready():
        raise HTTPException(status_code=503, detail="Entity extractor model not loaded")

    extraction_result = extractor.extract(request.text)

    nlp_analysis = {
        "intent": extraction_result["primary_intent"],
        "entities": {
            "tables": extraction_result["tables"],
            "time_filters": extraction_result["time_filters"],
        },
        "analysis_metadata": extraction_result["metadata"]
    }

    sql_result = sql_generator.generate_sql(nlp_analysis)

    if not sql_result.get("success", False):
        return {
            "success": False,
            "sql": None,
            "error": sql_result.get("error", "SQL generation failed"),
            "intent": extraction_result["primary_intent"]["type"] if extraction_result["primary_intent"] else None,
            "table": extraction_result["tables"][0]["table"] if extraction_result["tables"] else None,
            "metadata": extraction_result["metadata"]
        }

    return {
        "success": True,
        "sql": sql_result["sql"],
        "error": None,
        "intent": sql_result["intent"],
        "table": sql_result["table"],
        "metadata": sql_result["metadata"]
    }
