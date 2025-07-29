class QueryTemplates:
    """
    Farklı sorgu tipleri için şablonlar içerir
    """

    def __init__(self):
        pass
    def min_template(self, table_name, min_columns, where_clause=None):
        """Generate MIN query template"""
        min_expressions = [f"MIN({col}) as min_{col}" for col in min_columns]
        select_clause = ", ".join(min_expressions)

        base_query = f"SELECT {select_clause} FROM {table_name}"
        if where_clause:
            base_query += f" WHERE {where_clause}"

        return base_query
    
    def max_template(self, table_name, max_columns, where_clause=None):
        """Generate MAX query template"""
        max_expressions = [f"MAX({col}) as max_{col}" for col in max_columns]
        select_clause = ", ".join(max_expressions)

        base_query = f"SELECT {select_clause} FROM {table_name}"
        if where_clause:
            base_query += f" WHERE {where_clause}"

        return base_query

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
    
    def join_template(self, 
                  left_table, 
                  right_table, 
                  join_condition, 
                  select_columns, 
                  where_clause=None, 
                  join_type="INNER", 
                  limit=50):
        """
        Generate JOIN query between two tables.
        """
        columns_str = ", ".join(select_columns)
        query = f"SELECT {columns_str} FROM {left_table} {join_type} JOIN {right_table} ON {join_condition}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        query += f" ORDER BY 1 LIMIT {limit}"
        return query


    