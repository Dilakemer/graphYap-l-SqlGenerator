# from .schema_mapper import SchemaMapper
from src.query_builder.schema_mapper import SchemaMapper

from src.query_builder.query_templates import QueryTemplates
from src.query_builder.query_validator import QueryValidator
from src.query_builder.relation_mapper import RelationMapper

#Bu yardımcı fonksiyon, NLP analizinden gelen varlıkları tarayarak "en fazla" (MAX) veya "en az" (MIN) gibi agregasyon modifikatörlerini tespit eder.
def extract_aggregation_modifier(entities):
    print(f"Debug: Processing entities - {entities}")  # Debug satırı
    for ent in entities:
        label = ent.get("label", "")
        text = ent.get("text", "")
        print(f"Debug: Checking entity - Label: {label}, Text: {text}")  # Debug satırı
        
        if "INTENT_MAX" in label or "en fazla" in text.lower():
            print("Debug: Found MAX intent")  # Debug satırı
            return "MAX"
        elif "INTENT_MIN" in label or "en az" in text.lower():
            print("Debug: Found MIN intent")  # Debug satırı
            return "MIN"
    return None

class SQLGenerator:
    """
    Main SQL Generator that orchestrates query building
    Converts NLP analysis results to PostgreSQL queries
    Optimized without unnecessary table mapping
    """

    def __init__(self):
        self.schema_mapper = SchemaMapper()
        self.query_templates = QueryTemplates()
        self.validator = QueryValidator()
        self.relation_mapper = RelationMapper()
        # Statistics
        self.queries_generated = 0
        self.successful_generations = 0
    
    
    def generate_sql(self, nlp_analysis):
        self.queries_generated += 1
        try:
            # 1. Girdi validasyonu
            if not self._validate_input(nlp_analysis):
                return {"success": False, "error": "Invalid NLP analysis input", "sql": None,
                        "debug_info": self._get_debug_info(nlp_analysis)}

            if not nlp_analysis.get("analysis_metadata", {}).get("sql_ready", False):
                return {"success": False, "error": "NLP analysis not ready for SQL generation", "sql": None,
                        "debug_info": self._get_debug_info(nlp_analysis)}

            intent = nlp_analysis["intent"]["type"]
            entities = nlp_analysis["entities"]
            tables = entities["tables"]
            time_filters = entities.get("time_filters", [])
            filters = entities.get("filters", [])

            # 2. aggregation_modifier'ın raw intent/entity listesinden çıkarımı
            if "aggregation_modifier" not in entities:
                raw_ents = entities.get("entities", [])#min max içermeyen entity ler
    
                entities["aggregation_modifier"] = extract_aggregation_modifier(raw_ents)

            # 3. Alias ve join path hazırlığı tablolara alias atama t0 ve t1 gibi
            main_table = tables[0]["table"]
            join_clauses = []
            used_joins = set()
            aliases = {}
            alias_counter = 0
            used_aliases = set()

            def assign_alias(table_name):
                nonlocal alias_counter
                if table_name not in aliases:
                    alias = f"t{alias_counter}"
                    while alias in used_aliases:
                        alias_counter += 1
                        alias = f"t{alias_counter}"
                    aliases[table_name] = alias
                    used_aliases.add(alias)
                    alias_counter += 1
                return aliases[table_name]

            assign_alias(main_table)
            for entry in tables[1:]:
                assign_alias(entry["table"])
                path = self.relation_mapper.get_join_paths(main_table, entry["table"])
                if not path:
                    return {"success": False,
                            "error": f"No join path found between {main_table} and {entry['table']}",
                            "sql": None}
                for f_table, f_col, t_table, t_col in path:
                    sig = (f_table, f_col, t_table, t_col)
                    if sig in used_joins:
                        continue
                    used_joins.add(sig)
                    fa, ta = assign_alias(f_table), assign_alias(t_table)
                    join_clauses.append(
                        f"JOIN {t_table} {ta} ON {fa}.{f_col} = {ta}.{t_col}"
                    )

            where_clause = self.build_where_clause(filters, time_filters, aliases)

            # 4. Intent’e göre SQL oluşturma
            if intent == "SELECT":
                sql = self._generate_select_multi_table(tables, aliases, join_clauses, where_clause)

            elif intent == "COUNT":
                agg_mod = entities.get("aggregation_modifier")
                if not agg_mod:
                    lbl = nlp_analysis["intent"].get("label", "").lower()
                    if "en fazla" in lbl or "en çok" in lbl:
                        agg_mod = "MAX"
                    elif "en az" in lbl or "en düşük" in lbl:
                        agg_mod = "MIN"
                order_by = "DESC" if agg_mod == "MAX" else "ASC" if agg_mod == "MIN" else None
                limit = 1 if agg_mod in ("MAX", "MIN") else None
                sql = self._generate_count_multi_table(
                    tables, aliases, join_clauses, where_clause,
                    order_by=order_by, limit=limit
                )

            elif intent == "SUM":
                agg_mod = entities.get("aggregation_modifier")
                if not agg_mod:
                    lbl = nlp_analysis["intent"].get("label", "").lower()
                    if "en fazla" in lbl or "en çok" in lbl:
                        agg_mod = "MAX"
                    elif "en az" in lbl or "en düşük" in lbl:
                        agg_mod = "MIN"
                order_by = "DESC" if agg_mod == "MAX" else "ASC" if agg_mod == "MIN" else None
                limit = 1 if agg_mod in ("MAX", "MIN") else None
                sql = self._generate_sum_multi_table(
                    tables, aliases, join_clauses, where_clause,
                    order_by=order_by, limit=limit
                )

            elif intent == "AVG":
                sql = self._generate_avg_multi_table(tables, aliases, join_clauses, where_clause)

            elif intent == "AGGREGATE":
                func = nlp_analysis["intent"].get("function", "").upper()
                col = nlp_analysis["intent"].get("target_column")
                alias = assign_alias(main_table)
                if func in ("MIN", "MAX") and col:
                    sql = f"SELECT {func}({alias}.{col}) FROM {main_table} {alias}"
                    if where_clause:
                        sql += f" WHERE {where_clause}"
                else:
                    return {"success": False, "error": "Aggregate function or target column missing", "sql": None}

            else:
                return {"success": False, "error": f"Unsupported intent: {intent}", "sql": None}

            # 5. Validator ile son kontrol
            valid, err = self.validator.validate(sql)
            if not valid:
                return {"success": False, "error": f"Generated SQL failed validation: {err}", "sql": sql}

            self.successful_generations += 1
            return {
                "success": True,
                "sql": sql,
                "intent": intent,
                "tables": [t["table"] for t in tables],
                "has_time_filter": bool(time_filters),
                "confidence": nlp_analysis["intent"].get("confidence"),
                "metadata": {
                    "query_type": intent.lower(),
                    "complexity": "medium" if len(tables) > 1 else "simple",
                    "table_info": [
                        self.schema_mapper.get_table_info(t["table"])
                        for t in tables
                    ]
                }
            }

        except Exception as e:
            return {"success": False, "error": f"SQL generation failed: {e}", "sql": None,
                    "exception_type": type(e).__name__}


            
    def _generate_avg_multi_table(self, tables, aliases, join_clauses, where_clause):
        # En uygun tabloyu bulmak için toplam miktar sütunu olan tabloyu ara
        target_table = None
        target_column = None

        for table in tables:
            table_name = table["table"]
            schema = self.schema_mapper.get_table_schema(table_name)
            if "total_amount" in schema.get("avg_columns", []):
                target_table = table_name
                target_column = "total_amount"
                break

        if not target_table:
            # Varsayılan fallback: ilk tablo ve onun avg_columns'undan birini kullan
            target_table = tables[0]["table"]
            schema = self.schema_mapper.get_table_schema(target_table)
            avg_cols = schema.get("avg_columns", [])
            if avg_cols:
                target_column = avg_cols[0]
            else:
                raise ValueError(f"No avg columns found in table {target_table}")

        alias = aliases[target_table]
        join_part = " ".join(join_clauses) if join_clauses else ""
        where_part = f" WHERE {where_clause}" if where_clause else ""

        sql = (
            f"SELECT AVG({alias}.{target_column}) AS average_amount "
            f"FROM {target_table} {alias} "
            f"{join_part} "
            f"{where_part}"
        )
        return sql


        
    def _generate_avg_multi_table(self, tables, aliases, join_clauses, where_clause):
        target_table = None
        target_column = None

        for table in tables:
            table_name = table["table"]
            schema = self.schema_mapper.get_table_schema(table_name)
            if "avg_columns" in schema and schema["avg_columns"]:
                target_table = table_name
                target_column = schema["avg_columns"][0]
                break

        if not target_table:
            # Fallback: ilk tablonun herhangi bir avg_column'u
            target_table = tables[0]["table"]
            schema = self.schema_mapper.get_table_schema(target_table)
            avg_cols = schema.get("avg_columns", [])
            if avg_cols:
                target_column = avg_cols[0]
            else:
                raise ValueError(f"No avg columns found in table {target_table}")

        alias = aliases[target_table]
        join_part = " ".join(join_clauses) if join_clauses else ""
        where_part = f" WHERE {where_clause}" if where_clause else ""

        sql = (
            f"SELECT AVG({alias}.{target_column}) AS average_amount "
            f"FROM {target_table} {alias} "
            f"{join_part} "
            f"{where_part}"
        )
        return sql


    def _generate_sum_multi_table(self, tables, aliases, join_clauses, where_clause, order_by=None, limit=None):
        target_table = None
        target_column = None

        for table in reversed(tables):  # Genelde son tablo sum sütunu içerebilir
            table_name = table["table"]
            schema = self.schema_mapper.get_table_schema(table_name)
            if "sum_columns" in schema and schema["sum_columns"]:
                target_table = table_name
                target_column = schema["sum_columns"][0]
                break

        if not target_table:
            # Fallback
            target_table = tables[-1]["table"]
            schema = self.schema_mapper.get_table_schema(target_table)
            sum_cols = schema.get("sum_columns", [])
            if sum_cols:
                target_column = sum_cols[0]
            else:
                raise ValueError(f"No sum columns found in table {target_table}")

        main_table = tables[0]["table"]
        ma = aliases[main_table]
        la = aliases[target_table]
        group_col = self.schema_mapper.get_table_schema(main_table).get("display_columns", ["*"])[0]

        select = f"SELECT {ma}.{group_col} AS group_field, SUM({la}.{target_column}) AS sum_{target_column}"
        join_p = " ".join(join_clauses)
        where_p = f"WHERE {where_clause}" if where_clause else ""
        group_by = f"GROUP BY {ma}.{group_col}"
        order_p = f" ORDER BY sum_{target_column} {order_by}" if order_by else ""
        limit_p = f" LIMIT {limit}" if limit else ""

        sql = f"{select} FROM {main_table} {ma} {join_p} {where_p} {group_by}{order_p}{limit_p}"
        return sql


    def _generate_count_multi_table(self, tables, aliases, join_clauses, where_clause, order_by=None, limit=None):
        main_table = tables[0]["table"]
        ma = aliases[main_table]
        join_p = " ".join(join_clauses)
        where_p = f"WHERE {where_clause}" if where_clause else ""
        group_col = self.schema_mapper.get_table_schema(main_table).get("display_columns", ["*"])[0]

        sql = (f"SELECT {ma}.{group_col} AS group_field, COUNT(*) AS total_count "
            f"FROM {main_table} {ma} {join_p} {where_p} "
            f"GROUP BY {ma}.{group_col}")
        if order_by: sql += f" ORDER BY total_count {order_by}"
        if limit:    sql += f" LIMIT {limit}"
        return sql



    def _validate_input(self, nlp_analysis):
        """Validate NLP analysis input structure"""
        if not isinstance(nlp_analysis, dict):
            return False

        required_keys = ["intent", "entities", "analysis_metadata"]
        for key in required_keys:
            if key not in nlp_analysis:
                return False

        # Check intent structure
        intent = nlp_analysis.get("intent", {})
        if not isinstance(intent, dict) or "type" not in intent or "confidence" not in intent:
            return False

        # Check entities structure
        entities = nlp_analysis.get("entities", {})
        if not isinstance(entities, dict) or "tables" not in entities:
            return False

        # Must have at least one table
        tables = entities.get("tables", [])
        if not isinstance(tables, list) or len(tables) == 0:
            return False

        # First table must have required structure
        first_table = tables[0]
        if not isinstance(first_table, dict) or "table" not in first_table:
            return False

        return True

    def _generate_by_intent(self, intent, table_name, table_schema, where_clause):
        """Generate SQL based on intent type"""
        intent_generators = {
            "SELECT": self._generate_select,
            "COUNT": self._generate_count,
            "SUM": self._generate_sum,
            "AVG": self._generate_avg
        }

        generator = intent_generators.get(intent)
        if not generator:
            raise ValueError(f"Unsupported intent: {intent}")

        return generator(table_name, table_schema, where_clause)

    def _generate_select(self, table_name, table_schema, where_clause):
        """Generate SELECT query"""
        columns = table_schema.get("display_columns", ["*"])
        return self.query_templates.select_template(table_name, columns, where_clause)

    def _generate_count(self, table_name, table_schema, where_clause):
        """Generate COUNT query"""
        count_column = table_schema.get("countable_column", "*")
        return self.query_templates.count_template(table_name, count_column, where_clause)

    def _generate_sum(self, table_name, table_schema, where_clause):
        """Generate SUM query"""
        sum_columns = table_schema.get("sum_columns", [])
        if not sum_columns:
            # Fallback to count if no summable columns
            return self._generate_count(table_name, table_schema, where_clause)

        return self.query_templates.sum_template(table_name, sum_columns, where_clause)

    def _generate_avg(self, table_name, table_schema, where_clause):
        """Generate AVG query"""
        avg_columns = table_schema.get("avg_columns", [])
        if not avg_columns:
            # Fallback to count if no averageable columns
            return self._generate_count(table_name, table_schema, where_clause)

        return self.query_templates.avg_template(table_name, avg_columns, where_clause)

    def _get_debug_info(self, nlp_analysis):
        """Get debug information for troubleshooting"""
        debug_info = {
            "input_type": type(nlp_analysis).__name__,
            "available_tables": self.schema_mapper.get_all_tables()
        }

        if isinstance(nlp_analysis, dict):
            debug_info.update({
                "input_keys": list(nlp_analysis.keys()),
                "intent": nlp_analysis.get("intent", "Missing"),
                "entities": nlp_analysis.get("entities", "Missing"),
                "sql_ready": nlp_analysis.get("analysis_metadata", {}).get("sql_ready", "Missing")
            })
        else:
            debug_info["error"] = "Input is not a dictionary"

        return debug_info

    def get_statistics(self):
        """Get generation statistics"""
        success_rate = (self.successful_generations / self.queries_generated * 100) if self.queries_generated > 0 else 0

        return {
            "total_queries": self.queries_generated,
            "successful_queries": self.successful_generations,
            "failed_queries": self.queries_generated - self.successful_generations,
            "success_rate": round(success_rate, 2),
            "available_tables": len(self.schema_mapper.get_all_tables())
        }

    def get_supported_features(self):
        """Get supported features information"""
        return {
            "supported_intents": ["SELECT", "COUNT", "SUM", "AVG"],
            "supported_tables": self.schema_mapper.get_all_tables(),
            "time_filters": ["current_month", "current_year", "last_month", "last_year", "today", "last_week","current_week"],
            "total_table_count": len(self.schema_mapper.get_all_tables())
        }

    def test_schema_compatibility(self):
        """Test schema compatibility and return diagnostics"""
        results = {}

        for table_name in self.schema_mapper.get_all_tables():
            table_info = self.schema_mapper.get_table_info(table_name)
            results[table_name] = {
                "schema_valid": table_info is not None,
                "has_display_columns": table_info and table_info["display_column_count"] > 0,
                "supports_aggregation": table_info and (table_info["supports_sum"] or table_info["supports_avg"]),
                "has_date_column": table_info and table_info["date_column"] is not None
            }

        return results
    
    def _generate_select_multi_table(self, tables, aliases, join_clauses, where_clause):
        main_table = tables[0]["table"]
        base_columns = []
        for table in tables:
            table_name = table["table"]
            schema = self.schema_mapper.get_table_schema(table_name)
            alias = aliases[table_name]
            cols = schema.get("display_columns", ["*"])
            base_columns.extend([f"{alias}.{col}" for col in cols])

        sql = f"SELECT {', '.join(base_columns)} FROM {main_table} {aliases[main_table]} "
        if join_clauses:
            sql += " " + " ".join(join_clauses)
        if where_clause:
            sql += f" WHERE {where_clause}"

        return sql
    
    def build_time_filter(self, date_column, time_filter_obj):
        period = time_filter_obj.get("period")
        specific_date = time_filter_obj.get("date")

        filters = {
            "current_month": f"EXTRACT(MONTH FROM {date_column}) = EXTRACT(MONTH FROM CURRENT_DATE) AND EXTRACT(YEAR FROM {date_column}) = EXTRACT(YEAR FROM CURRENT_DATE)",
            "current_year": f"EXTRACT(YEAR FROM {date_column}) = EXTRACT(YEAR FROM CURRENT_DATE)",
            "last_month": f"{date_column} >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND {date_column} < DATE_TRUNC('month', CURRENT_DATE)",
            "last_year": f"EXTRACT(YEAR FROM {date_column}) = EXTRACT(YEAR FROM CURRENT_DATE) - 1",
            "today": f"DATE({date_column}) = CURRENT_DATE",
            "last_week": f"{date_column} >= DATE_TRUNC('week', CURRENT_DATE - INTERVAL '1 week') AND {date_column} < DATE_TRUNC('week', CURRENT_DATE)",
            "current_week": f"EXTRACT(WEEK FROM {date_column}) = EXTRACT(WEEK FROM CURRENT_DATE) AND EXTRACT(YEAR FROM {date_column}) = EXTRACT(YEAR FROM CURRENT_DATE)",
        }

        if period in filters:
            return filters[period]

        if period == "specific_date" and specific_date:
            return f"DATE({date_column}) = '{specific_date}'"
        
        if period == "year":
            start_date = time_filter_obj.get("start_date")
            end_date = time_filter_obj.get("end_date")
            if start_date and end_date:
                return f"{date_column} BETWEEN '{start_date}' AND '{end_date}'"


        return f"{date_column} >= CURRENT_DATE - INTERVAL '1 month'"

    def build_where_clause(self, filters, time_filters, aliases):
        where_clauses = []

        if time_filters:
            main_table = list(aliases.keys())[0]
            date_column = self.schema_mapper.get_table_schema(main_table).get("date_column")
            if date_column:
                alias = aliases[main_table]
                time_filter_clause = self.build_time_filter(f"{alias}.{date_column}", time_filters[0])
                if time_filter_clause:
                    where_clauses.append(time_filter_clause)

        for f in filters:
            col = f.get("column")
            op = f.get("operator", "=")
            val = f.get("value")

            main_table = list(aliases.keys())[0]
            alias = aliases[main_table]

            if isinstance(val, str):
                val_str = f"'{val}'"
            else:
                val_str = str(val)

            where_clauses.append(f"{alias}.{col} {op} {val_str}")

        if where_clauses:
            return " AND ".join(where_clauses)
        else:
            return None


def create_sql_generator():
    """Factory function to create SQL generator instance"""
    return SQLGenerator()
