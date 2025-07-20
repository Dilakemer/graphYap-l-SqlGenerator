class QueryValidator:
    """
    SQL query validation for security and correctness
    Prevents injection attacks and validates structure
    """

    def __init__(self):
        self.dangerous_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'INSERT', 'UPDATE',
            'CREATE', 'GRANT', 'REVOKE', '--', ';DELETE', ';DROP', ';UPDATE'
        ]

    def validate(self, sql_query):
        """
        Validate SQL query for security and structure

        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
        if not sql_query or not sql_query.strip():
            return False, "Empty SQL query"

        sql_upper = sql_query.upper().strip()

        # Must start with SELECT
        if not sql_upper.startswith('SELECT'):
            return False, "Only SELECT queries are allowed"

        # Check for dangerous keywords (whole words only)
        import re
        for keyword in self.dangerous_keywords:
            # Use word boundaries to avoid false positives like CURRENT_DATE containing "CREATE"
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, sql_upper):
                return False, f"Dangerous keyword detected: {keyword}"

        # Basic structure validation
        if 'FROM' not in sql_upper:
            return False, "Invalid SQL structure: missing FROM clause"

        # Check for basic SQL injection patterns (but allow CURRENT_DATE, EXTRACT, etc.)
        dangerous_patterns = [r"';", r'";', r'--', r'/\*', r'\*/']
        for pattern in dangerous_patterns:
            if re.search(pattern, sql_query):
                return False, f"Potentially unsafe pattern detected: {pattern}"

        return True, "Valid SQL query"

    def sanitize_table_name(self, table_name):
        """Sanitize table name to prevent injection"""
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', table_name)
        return sanitized.lower()