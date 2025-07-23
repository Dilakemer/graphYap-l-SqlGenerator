class RelationMapper:
    """
    self.relations: Foreign key bağlantılarını manuel olarak belirtiyoruz.

get_related_table(): Belirli bir kolonun bağlı olduğu tabloyu verir.

find_join_path(): Başlangıç ve hedef tablo verildiğinde JOIN sırasını çıkarır (örneğin order_details → orders → customers).
    """

    def __init__(self):
        # Format: (source_table, source_column) -> (target_table, target_column)
        self.relations = {
            ("products", "category_id"): ("categories", "id"),
            ("products", "supplier_id"): ("suppliers", "id"),
            ("orders", "customer_id"): ("customers", "id"),
            ("orders", "employee_id"): ("employees", "id"),
            ("order_details", "order_id"): ("orders", "id"),
            ("order_details", "product_id"): ("products", "id"),
            ("purchase_orders", "supplier_id"): ("suppliers", "id"),
            ("purchase_orders", "employee_id"): ("employees", "id"),
        }

    def get_related_table(self, source_table, source_column):
        """Returns the related (target_table, target_column) if exists"""
        return self.relations.get((source_table, source_column), None)

    def get_all_relations(self):
        """Returns all defined relations"""
        return self.relations

    def find_join_path(self, start_table, end_table, visited=None):
        if visited is None:
            visited = set()
        visited.add(start_table)

        for (src_table, src_col), (tgt_table, tgt_col) in self.relations.items():
            # İleri yön
            if src_table == start_table and tgt_table == end_table:
                return [(src_table, src_col, tgt_table, tgt_col)]

            if src_table == start_table and tgt_table not in visited:
                path = self.find_join_path(tgt_table, end_table, visited)
                if path:
                    return [(src_table, src_col, tgt_table, tgt_col)] + path

            # Geri yön (ters ilişkiyi de ara)
            if tgt_table == start_table and src_table == end_table:
                return [(tgt_table, tgt_col, src_table, src_col)]

            if tgt_table == start_table and src_table not in visited:
                path = self.find_join_path(src_table, end_table, visited)
                if path:
                    return [(tgt_table, tgt_col, src_table, src_col)] + path

        return None

    
    def get_join_paths(self, from_table, to_table):
        """
        from_table'dan to_table'a join path'ı DFS ile bulur.
        Liste olarak join adımlarını döner:
        [(from_table, from_col, to_table, to_col), ...]
        """
        return self.find_join_path(from_table, to_table)
