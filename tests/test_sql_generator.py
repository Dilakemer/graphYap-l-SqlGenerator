class SQLGenerator:
    def __init__(self):
        # Basit schema_mapper mock'u
        self.schema_mapper = self.DummySchemaMapper()

    class DummySchemaMapper:
        def get_table_schema(self, table_name):
            return {"date_column": "created_date"}

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


def test_sql_generator_time_filters():
    sql_gen = SQLGenerator()

    aliases = {"customers": "t0"}

    # Specific date testi
    time_filters = [{"period": "specific_date", "date": "2024-06-15"}]
    filters = []
    where_clause = sql_gen.build_where_clause(filters, time_filters, aliases)
    print("Specific date where clause:", where_clause)

    # Bilinen period testi
    time_filters = [{"period": "current_month"}]
    where_clause = sql_gen.build_where_clause(filters, time_filters, aliases)
    print("Current month where clause:", where_clause)

    # Bilinmeyen period fallback testi
    time_filters = [{"period": "unknown_period"}]
    where_clause = sql_gen.build_where_clause(filters, time_filters, aliases)
    print("Fallback where clause:", where_clause)

    # Genel filtre ile birlikte
    filters = [{"column": "city", "operator": "=", "value": "Istanbul"}]
    time_filters = [{"period": "specific_date", "date": "2024-07-01"}]
    where_clause = sql_gen.build_where_clause(filters, time_filters, aliases)
    print("Combined filters where clause:", where_clause)


if __name__ == "__main__":
    test_sql_generator_time_filters()
