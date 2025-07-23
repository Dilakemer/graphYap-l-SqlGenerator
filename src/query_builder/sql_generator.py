# from .schema_mapper import SchemaMapper
from src.query_builder.schema_mapper import SchemaMapper

from src.query_builder.query_templates import QueryTemplates
from src.query_builder.query_validator import QueryValidator
from src.query_builder.relation_mapper import RelationMapper
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
    
    def build_sql_with_joins(self, start_table, end_table):
        """
        start_table'dan end_table'a olan JOIN adımlarını kullanarak SQL JOIN bloklarını oluşturur.
        JOIN sırasını relation_mapper.find_join_path() ile bulur.
        """

        join_path = self.relation_mapper.find_join_path(start_table, end_table)

        if not join_path:
            print(f"❌ SQL Error: No join path found between {start_table} and {end_table}")
            return None

        alias_map = {}
        alias_counter = 0
        used_aliases = set()

        # İlk tabloyu alias'la başlat
        if start_table not in alias_map:
            alias_map[start_table] = f"t{alias_counter}"
            used_aliases.add(f"{start_table}::{alias_map[start_table]}")
            alias_counter += 1

        sql = f"FROM {start_table} {alias_map[start_table]}\n"

        print("🔍 JOIN path bulundu:")
        for step in join_path:
            print("   ", step)

        for src, src_col, tgt, tgt_col in join_path:
            if not src or not tgt or not src_col or not tgt_col:
                print(f"❌ Geçersiz JOIN adımı: {src}.{src_col} → {tgt}.{tgt_col}")
                return None

            # Alias oluştururken tablo+alias kombinasyonunu iki kez vermemeye dikkat et
            if src not in alias_map:
                alias_name = f"t{alias_counter}"
                while f"{src}::{alias_name}" in used_aliases:
                    alias_counter += 1
                    alias_name = f"t{alias_counter}"
                alias_map[src] = alias_name
                used_aliases.add(f"{src}::{alias_name}")
                alias_counter += 1

            if tgt not in alias_map:
                alias_name = f"t{alias_counter}"
                while f"{tgt}::{alias_name}" in used_aliases:
                    alias_counter += 1
                    alias_name = f"t{alias_counter}"
                alias_map[tgt] = alias_name
                used_aliases.add(f"{tgt}::{alias_name}")
                alias_counter += 1

            src_alias = alias_map[src]
            tgt_alias = alias_map[tgt]

            sql += f"JOIN {tgt} {tgt_alias} ON {src_alias}.{src_col} = {tgt_alias}.{tgt_col}\n"

        return sql



    def generate_sql(self, nlp_analysis):
        self.queries_generated += 1

        try:
            if not self._validate_input(nlp_analysis):
                return {
                    "success": False,
                    "error": "Invalid NLP analysis input",
                    "sql": None,
                    "debug_info": self._get_debug_info(nlp_analysis)
                }

            if not nlp_analysis.get("analysis_metadata", {}).get("sql_ready", False):
                return {
                    "success": False,
                    "error": "NLP analysis not ready for SQL generation",
                    "sql": None,
                    "debug_info": self._get_debug_info(nlp_analysis)
                }

            intent = nlp_analysis["intent"]["type"]
            tables = nlp_analysis["entities"]["tables"]
            time_filters = nlp_analysis["entities"].get("time_filters", [])

            main_table = tables[0]["table"]

            join_clauses = []
            aliases = {main_table: "t0"}
            alias_counter = 1

            # Her JOIN adımındaki tüm tablolar için alias ataması yapıyoruz
            for table_entry in tables[1:]:
                current_table = table_entry["table"]
                if current_table not in aliases:
                    aliases[current_table] = f"t{alias_counter}"
                    alias_counter += 1

                join_path = self.relation_mapper.get_join_paths(main_table, current_table)
                if not join_path:
                    return {
                        "success": False,
                        "error": f"No join path found between {main_table} and {current_table}",
                        "sql": None
                    }

                for from_table, from_col, to_table, to_col in join_path:
                    # Ara tablolara alias ataması
                    if from_table not in aliases:
                        aliases[from_table] = f"t{alias_counter}"
                        alias_counter += 1
                    if to_table not in aliases:
                        aliases[to_table] = f"t{alias_counter}"
                        alias_counter += 1

                    from_alias = aliases[from_table]
                    to_alias   = aliases[to_table]

                    join_clause = (
                        f"JOIN {to_table} {to_alias} "
                        f"ON {from_alias}.{from_col} = {to_alias}.{to_col}"
                    )
                    if join_clause not in join_clauses:
                        join_clauses.append(join_clause)

            print("💡 Join clauses constructed:")
            for jc in join_clauses:
                print(f"   {jc}")

            where_clause = None
            if time_filters:
                time_period = time_filters[0]["period"]
                date_column = self.schema_mapper.get_table_schema(main_table).get("date_column")
                where_clause = self.query_templates.build_time_filter(
                    f"{aliases[main_table]}.{date_column}",
                    time_period
                )

            if intent == "SELECT":
                sql = self._generate_select_multi_table(tables, aliases, join_clauses, where_clause)
            elif intent == "COUNT":
                sql = self._generate_count_multi_table(tables, aliases, join_clauses, where_clause)
            elif intent == "SUM":
                sql = self._generate_sum_multi_table(tables, aliases, join_clauses, where_clause)
            elif intent == "AVG":
                sql = self._generate_avg_multi_table(tables, aliases, join_clauses, where_clause)

            else:
                return {
                    "success": False,
                    "error": f"Unsupported intent: {intent}",
                    "sql": None
                }

            is_valid, validation_error = self.validator.validate(sql)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Generated SQL failed validation: {validation_error}",
                    "sql": sql
                }

            self.successful_generations += 1

            return {
                "success": True,
                "sql": sql,
                "intent": intent,
                "tables": [t["table"] for t in tables],
                "has_time_filter": len(time_filters) > 0,
                "confidence": nlp_analysis["intent"]["confidence"],
                "metadata": {
                    "query_type": intent.lower(),
                    "complexity": "medium" if len(tables) > 1 else "simple",
                    "table_info": [self.schema_mapper.get_table_info(t["table"]) for t in tables]
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"SQL generation failed: {str(e)}",
                "sql": None,
                "exception_type": type(e).__name__
            }
        
    def _generate_count_multi_table(self, tables, aliases, join_clauses, where_clause):
        """
        Çoklu tabloya dayalı COUNT sorgusu üretir.
        Örnek: 'önceki 3 yıl siparişleri kaç'
        """
        main_table = tables[0]["table"]
        main_alias = aliases[main_table]

        # JOIN parçaları
        join_part = " ".join(join_clauses) if join_clauses else ""

        # WHERE filtresi
        where_part = f" WHERE {where_clause}" if where_clause else ""

        # COUNT(*) olarak döner
        sql = (
            f"SELECT COUNT(*) AS total_count "
            f"FROM {main_table} {main_alias} "
            f"{join_part}"
            f"{where_part};"
        )
        return sql
    
    def _generate_avg_multi_table(self, tables, aliases, join_clauses, where_clause):
        """
        Çoklu tabloya dayalı AVG sorgusu üretir.
        Örnek: 'bu yılın sipariş ortalaması'
        """
        main_table = tables[0]["table"]
        main_alias = aliases[main_table]

        # Varsayılan olarak 'total_amount' gibi bir sütun varsayabiliriz, istersen bunu dinamik hale getirebiliriz
        avg_column = "total_amount"

        # JOIN parçaları
        join_part = " ".join(join_clauses) if join_clauses else ""

        # WHERE filtresi
        where_part = f" WHERE {where_clause}" if where_clause else ""

        sql = (
            f"SELECT AVG({main_alias}.{avg_column}) AS average_amount "
            f"FROM {main_table} {main_alias} "
            f"{join_part}"
            f"{where_part};"
        )
        return sql

        
    def _generate_sum_multi_table(self, tables, aliases, join_clauses, where_clause):
        """
        Çoklu tabloya dayalı SUM sorgusu üretir.
        Örnek: 'Ürün kategori bilgisi ile sipariş tutarlarını getir'
        """
        # Burada hangi tablo/kolonda sum yapılacağını seç
        # Örnek: son tabloda 'total_price' kolonu olduğunu varsayıyoruz
        last_table = tables[-1]["table"]
        schema = self.schema_mapper.get_table_schema(last_table)
        sum_columns = schema.get("sum_columns", [])
        if not sum_columns:
            # Eğer schema'da tanımlı değilse hata
            raise ValueError(f"No sum_columns defined for table {last_table}")

        # Birden çok sum kolonu olabilir, biz ilkiyle ilerleyelim
        sum_col = sum_columns[0]
        main_table = tables[0]["table"]
        main_alias = aliases[main_table]
        last_alias = aliases[last_table]

        # SELECT kısmı: grup sütunları + SUM
        # Grup by için önceki tablonun ilk display column'unu alıyoruz örnek olarak
        group_col = self.schema_mapper.get_table_schema(main_table).get("display_columns", ["*"])[0]
        select_part = (
            f"SELECT {main_alias}.{group_col} AS group_field, "
            f"SUM({last_alias}.{sum_col}) AS sum_{sum_col}"
        )

        # Birleştirmeler
        join_part = " ".join(join_clauses)

        # GROUP BY
        group_by = f"GROUP BY {main_alias}.{group_col}"

        # WHERE
        where_part = f"WHERE {where_clause}" if where_clause else ""

        sql = f"{select_part} FROM {main_table} {main_alias} {join_part} {where_part} {group_by};"
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



def create_sql_generator():
    """Factory function to create SQL generator instance"""
    return SQLGenerator()