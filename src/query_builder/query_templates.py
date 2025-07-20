class QueryTemplates:
    """
    SQL query templates for different intents
    Generates PostgreSQL compatible queries
    """

    def __init__(self):
        pass

    def select_template(self, table_name, columns, where_clause=None, limit=50):
        """Generate SELECT query template"""
        columns_str = ", ".join(columns)
        base_query = f"SELECT {columns_str} FROM {table_name}"

        if where_clause:
            base_query += f" WHERE {where_clause}"

        base_query += f" ORDER BY id LIMIT {limit}"
        return base_query

    def count_template(self, table_name, count_column, where_clause=None):
        """Generate COUNT query template"""
        base_query = f"SELECT COUNT({count_column}) as total_count FROM {table_name}"

        if where_clause:
            base_query += f" WHERE {where_clause}"

        return base_query

    def sum_template(self, table_name, sum_columns, where_clause=None):
        """Generate SUM query template"""
        sum_expressions = [f"SUM({col}) as total_{col}" for col in sum_columns]
        select_clause = ", ".join(sum_expressions)

        base_query = f"SELECT {select_clause} FROM {table_name}"

        if where_clause:
            base_query += f" WHERE {where_clause}"

        return base_query

    def avg_template(self, table_name, avg_columns, where_clause=None):
        """Generate AVG query template"""
        avg_expressions = [f"ROUND(AVG({col}), 2) as avg_{col}" for col in avg_columns]
        select_clause = ", ".join(avg_expressions)

        base_query = f"SELECT {select_clause} FROM {table_name}"

        if where_clause:
            base_query += f" WHERE {where_clause}"

        return base_query

    def build_time_filter(self, date_column, time_period):
        """Build PostgreSQL time filter WHERE clause"""
        filters = {
            "current_month": f"EXTRACT(MONTH FROM {date_column}) = EXTRACT(MONTH FROM CURRENT_DATE) AND EXTRACT(YEAR FROM {date_column}) = EXTRACT(YEAR FROM CURRENT_DATE)",
            "current_year": f"EXTRACT(YEAR FROM {date_column}) = EXTRACT(YEAR FROM CURRENT_DATE)",
            "last_month": f"{date_column} >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND {date_column} < DATE_TRUNC('month', CURRENT_DATE)",
            "last_year": f"EXTRACT(YEAR FROM {date_column}) = EXTRACT(YEAR FROM CURRENT_DATE) - 1",
            "today": f"DATE({date_column}) = CURRENT_DATE",
            "last_week": f"{date_column} >= DATE_TRUNC('week', CURRENT_DATE - INTERVAL '1 week') AND {date_column} < DATE_TRUNC('week', CURRENT_DATE)",
            "current_week": f"EXTRACT(WEEK FROM {date_column}) = EXTRACT(WEEK FROM CURRENT_DATE) AND EXTRACT(YEAR FROM {date_column}) = EXTRACT(YEAR FROM CURRENT_DATE)"
        }

        return filters.get(time_period, f"{date_column} >= CURRENT_DATE - INTERVAL '1 month'")

