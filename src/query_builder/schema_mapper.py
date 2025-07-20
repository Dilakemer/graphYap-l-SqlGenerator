class SchemaMapper:
    """
    Database schema mapper for SQL generation
    Maps table names to their schema definitions
    No Turkish-English mapping needed (Entity Extractor handles that)
    """

    def __init__(self):
        # Database schema definition - matches actual PostgreSQL schema
        self.schema = {
            "customers": {
                "primary_key": "id",
                "date_column": "created_date",
                "display_columns": ["company_name", "contact_person", "city"],
                "countable_column": "id"
            },
            "products": {
                "primary_key": "id",
                "date_column": "created_date",
                "display_columns": ["product_name", "unit_price", "stock_quantity", "unit"],
                "countable_column": "id",
                "sum_columns": ["unit_price", "stock_quantity"],
                "avg_columns": ["unit_price", "stock_quantity"]
            },
            "orders": {
                "primary_key": "id",
                "date_column": "order_date",
                "display_columns": ["id", "order_date", "total_amount", "status"],
                "countable_column": "id",
                "sum_columns": ["total_amount"],
                "avg_columns": ["total_amount"]
            },
            "categories": {
                "primary_key": "id",
                "date_column": "created_date",
                "display_columns": ["category_name", "description"],
                "countable_column": "id"
            },
            "suppliers": {
                "primary_key": "id",
                "date_column": "created_date",
                "display_columns": ["company_name", "contact_person", "city"],
                "countable_column": "id"
            },
            "employees": {
                "primary_key": "id",
                "date_column": "hire_date",
                "display_columns": ["first_name", "last_name", "department", "position"],
                "countable_column": "id",
                "sum_columns": ["salary"],
                "avg_columns": ["salary"]
            },
            "order_details": {
                "primary_key": "id",
                "date_column": "order_date",  # Will need JOIN for time filters
                "display_columns": ["order_id", "product_id", "quantity", "unit_price", "total_price"],
                "countable_column": "id",
                "sum_columns": ["quantity", "unit_price", "total_price"],
                "avg_columns": ["quantity", "unit_price", "total_price"]
            },
            "purchase_orders": {
                "primary_key": "id",
                "date_column": "order_date",
                "display_columns": ["id", "order_date", "total_amount", "status", "delivery_date"],
                "countable_column": "id",
                "sum_columns": ["total_amount"],
                "avg_columns": ["total_amount"]
            }
        }

    def get_table_schema(self, table_name):
        """Get schema for table"""
        return self.schema.get(table_name, {})

    def is_valid_table(self, table_name):
        """Check if table exists in schema"""
        return table_name in self.schema

    def get_all_tables(self):
        """Get all available tables"""
        return list(self.schema.keys())

    def get_table_info(self, table_name):
        """Get detailed table information"""
        schema = self.get_table_schema(table_name)
        if not schema:
            return None

        return {
            "table_name": table_name,
            "exists": True,
            "primary_key": schema.get("primary_key"),
            "date_column": schema.get("date_column"),
            "supports_sum": bool(schema.get("sum_columns")),
            "supports_avg": bool(schema.get("avg_columns")),
            "display_column_count": len(schema.get("display_columns", []))
        }