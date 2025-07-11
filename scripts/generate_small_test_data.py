from faker import Faker
import random

# Turkish locale
fake = Faker("tr_TR")


def generate_small_test_data():
    """
    Küçük test data - PgAdmin'de hızlı çalışır:
    - 8 categories
    - 6 employees
    - 20 suppliers
    - 50 customers
    - 100 products
    - 150 orders
    - 300 order_details
    """

    sql_statements = []

    # 1. CATEGORIES (8 adet)
    categories_sql = """INSERT INTO categories (category_name, description) VALUES
('Elektronik', 'Bilgisayar, telefon ve elektronik ürünler'),
('Gıda', 'Gıda ve içecek ürünleri'),
('Tekstil', 'Giyim ve kumaş ürünleri'),
('Kozmetik', 'Güzellik ve bakım ürünleri'),
('Mobilya', 'Ev ve ofis mobilyaları'),
('Kitap', 'Kitap ve kırtasiye ürünleri'),
('Spor', 'Spor malzemeleri ve giyim'),
('Oyuncak', 'Çocuk oyuncakları');

"""
    sql_statements.append(categories_sql)

    # 2. EMPLOYEES (6 adet)
    employees_sql = """INSERT INTO employees (first_name, last_name, department, position, salary, hire_date, email) VALUES
('Emre', 'Çetin', 'Satış', 'Satış Temsilcisi', 8500.00, '2023-01-15', 'emre@firma.com'),
('Cansu', 'Aydın', 'Satış', 'Satış Müdürü', 12000.00, '2022-06-10', 'cansu@firma.com'),
('Oğuz', 'Karaca', 'Muhasebe', 'Muhasebeci', 9000.00, '2023-03-20', 'oguz@firma.com'),
('İrem', 'Polat', 'İnsan Kaynakları', 'İK Uzmanı', 8000.00, '2023-07-05', 'irem@firma.com'),
('Berk', 'Yücel', 'Lojistik', 'Depo Sorumlusu', 7500.00, '2023-09-12', 'berk@firma.com'),
('Deniz', 'Özer', 'Satış', 'Satış Temsilcisi', 8200.00, '2024-01-08', 'deniz@firma.com');

"""
    sql_statements.append(employees_sql)

    # 3. SUPPLIERS (20 adet - küçük!)
    company_suffixes = ["A.Ş.", "Ltd.", "Ltd. Şti."]
    suppliers_sql = "INSERT INTO suppliers (company_name, contact_person, email, phone, city) VALUES\n"
    supplier_values = []

    for i in range(20):
        company = f"{fake.company()} {random.choice(company_suffixes)}"
        person = fake.name()
        email = fake.email()
        phone = fake.phone_number()[:45]
        city = fake.city()
        supplier_values.append(
            f"('{company[:295]}', '{person}', '{email[:145]}', '{phone}', '{city}')"
        )

    suppliers_sql += ",\n".join(supplier_values) + ";\n\n"
    sql_statements.append(suppliers_sql)

    # 4. CUSTOMERS (50 adet - küçük!)
    customers_sql = "INSERT INTO customers (company_name, contact_person, email, phone, city) VALUES\n"
    customer_values = []

    for i in range(50):
        company = f"{fake.company()} Market"
        person = fake.name()
        email = fake.email()
        phone = fake.phone_number()[:45]
        city = fake.city()
        customer_values.append(
            f"('{company[:295]}', '{person}', '{email[:145]}', '{phone}', '{city}')"
        )

    customers_sql += ",\n".join(customer_values) + ";\n\n"
    sql_statements.append(customers_sql)

    # 5. PRODUCTS (100 adet - küçük!)
    product_names = [
        "iPhone",
        "Samsung",
        "Domates",
        "Ekmek",
        "Jean",
        "Parfüm",
        "Masa",
        "Kitap",
    ]
    products_sql = "INSERT INTO products (product_name, category_id, supplier_id, unit_price, unit, stock_quantity) VALUES\n"
    product_values = []
    units = ["adet", "kg", "litre"]

    for i in range(100):
        name = f"{random.choice(product_names)} {fake.word().title()}"
        category_id = random.randint(1, 8)  # 1-8 categories
        supplier_id = random.randint(1, 20)  # 1-20 suppliers
        price = round(random.uniform(10.0, 1000.0), 2)
        unit = random.choice(units)
        stock = random.randint(10, 200)
        product_values.append(
            f"('{name[:195]}', {category_id}, {supplier_id}, {price}, '{unit}', {stock})"
        )

    products_sql += ",\n".join(product_values) + ";\n\n"
    sql_statements.append(products_sql)

    # 6. ORDERS (150 adet - küçük!)
    orders_sql = "INSERT INTO orders (customer_id, employee_id, order_date, total_amount, status) VALUES\n"
    order_values = []
    statuses = ["completed", "pending", "shipping"]

    for i in range(150):
        customer_id = random.randint(1, 50)  # 1-50 customers
        employee_id = random.randint(1, 6)  # 1-6 employees
        order_date = fake.date_between(start_date="-6m", end_date="today")
        total_amount = round(random.uniform(100.0, 5000.0), 2)
        status = random.choice(statuses)
        order_values.append(
            f"({customer_id}, {employee_id}, '{order_date}', {total_amount}, '{status}')"
        )

    orders_sql += ",\n".join(order_values) + ";\n\n"
    sql_statements.append(orders_sql)

    # 7. ORDER_DETAILS (300 adet - küçük!)
    order_details_sql = "INSERT INTO order_details (order_id, product_id, quantity, unit_price, total_price) VALUES\n"
    detail_values = []

    for i in range(300):
        order_id = random.randint(1, 150)  # 1-150 orders
        product_id = random.randint(1, 100)  # 1-100 products
        quantity = random.randint(1, 10)
        unit_price = round(random.uniform(10.0, 500.0), 2)
        total_price = round(quantity * unit_price, 2)
        detail_values.append(
            f"({order_id}, {product_id}, {quantity}, {unit_price}, {total_price})"
        )

    order_details_sql += ",\n".join(detail_values) + ";\n\n"
    sql_statements.append(order_details_sql)

    return "\n".join(sql_statements)


if __name__ == "__main__":
    print("Generating small test data...")
    sql_content = generate_small_test_data()

    with open("data/small_test_data.sql", "w", encoding="utf-8") as f:
        f.write("-- SMALL TEST DATA for NLP-SQL Project\n")
        f.write("-- Quick testing with manageable data size\n\n")
        f.write(sql_content)
        f.write("\nSELECT 'Small test data generated successfully!' as result;")

    print("✅ small_test_data.sql created!")
