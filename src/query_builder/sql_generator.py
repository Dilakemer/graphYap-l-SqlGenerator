from schema_mapper import SchemaMapper
from query_templates import QueryTemplates
from query_validator import QueryValidator

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

        # Statistics
        self.queries_generated = 0
        self.successful_generations = 0

    def generate_sql(self, nlp_analysis):
        """
        Generate SQL query from NLP analysis result

        Args:
            nlp_analysis: Dictionary from NLPProcessor.analyze()

        Returns:
            Dictionary with SQL query and metadata
        """
        self.queries_generated += 1

        try:
            # Validate input
            if not self._validate_input(nlp_analysis):
                return {
                    "success": False,
                    "error": "Invalid NLP analysis input",
                    "sql": None,
                    "debug_info": self._get_debug_info(nlp_analysis)
                }

            # Check if SQL ready
            if not nlp_analysis.get("analysis_metadata", {}).get("sql_ready", False):
                return {
                    "success": False,
                    "error": "NLP analysis not ready for SQL generation",
                    "sql": None,
                    "debug_info": self._get_debug_info(nlp_analysis)
                }

            # Extract components
            intent = nlp_analysis["intent"]["type"]
            tables = nlp_analysis["entities"]["tables"]
            time_filters = nlp_analysis["entities"]["time_filters"]

            # Get table name directly (Entity Extractor already provides correct English name)
            table_name = tables[0]["table"]

            # Validate table exists in our schema
            if not self.schema_mapper.is_valid_table(table_name):
                return {
                    "success": False,
                    "error": f"Unknown table: {table_name}",
                    "sql": None,
                    "available_tables": self.schema_mapper.get_all_tables()
                }

            # Get table schema
            table_schema = self.schema_mapper.get_table_schema(table_name)

            # Build time filter if exists
            where_clause = None
            if time_filters:
                time_period = time_filters[0]["period"]
                date_column = table_schema["date_column"]
                where_clause = self.query_templates.build_time_filter(date_column, time_period)

            # Generate SQL based on intent
            sql_query = self._generate_by_intent(intent, table_name, table_schema, where_clause)

            # Validate generated SQL
            is_valid, validation_error = self.validator.validate(sql_query)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Generated SQL failed validation: {validation_error}",
                    "sql": sql_query
                }

            self.successful_generations += 1

            return {
                "success": True,
                "sql": sql_query,
                "intent": intent,
                "table": table_name,
                "has_time_filter": len(time_filters) > 0,
                "confidence": nlp_analysis["intent"]["confidence"],
                "metadata": {
                    "query_type": intent.lower(),
                    "complexity": "simple" if len(time_filters) <= 1 else "medium",
                    "table_info": self.schema_mapper.get_table_info(table_name)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"SQL generation failed: {str(e)}",
                "sql": None,
                "exception_type": type(e).__name__
            }

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
            "time_filters": ["current_month", "current_year", "last_month", "last_year", "today", "last_week",
                             "current_week"],
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


def create_sql_generator():
    """Factory function to create SQL generator instance"""
    return SQLGenerator()