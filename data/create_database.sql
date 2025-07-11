-- Turkish NLP-SQL Demo Database Schema
-- 8 Tables for ERP System

-- 1. CATEGORIES (Kategoriler)
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    description TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. SUPPLIERS (Tedarikçiler)
CREATE TABLE suppliers (
    id SERIAL PRIMARY KEY,
    company_name VARCHAR(200) NOT NULL,
    contact_person VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(50),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. CUSTOMERS (Müşteriler)
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    company_name VARCHAR(200) NOT NULL,
    contact_person VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(50),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. EMPLOYEES (Çalışanlar)
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    department VARCHAR(50),
    position VARCHAR(100),
    salary DECIMAL(10,2),
    hire_date DATE,
    email VARCHAR(100),
    phone VARCHAR(20)
);

-- 5. PRODUCTS (Ürünler)
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    category_id INTEGER REFERENCES categories(id),
    supplier_id INTEGER REFERENCES suppliers(id),
    unit_price DECIMAL(10,2),
    unit VARCHAR(20) DEFAULT 'adet',
    stock_quantity INTEGER DEFAULT 0,
    description TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. ORDERS (Müşteri Siparişleri)
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    employee_id INTEGER REFERENCES employees(id),
    order_date DATE DEFAULT CURRENT_DATE,
    total_amount DECIMAL(12,2),
    status VARCHAR(20) DEFAULT 'pending',
    notes TEXT
);

-- 7. ORDER_DETAILS (Sipariş Detayları)
CREATE TABLE order_details (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(12,2),
    CONSTRAINT check_quantity CHECK (quantity > 0)
);

-- 8. PURCHASE_ORDERS (Tedarikçi Alımları)
CREATE TABLE purchase_orders (
    id SERIAL PRIMARY KEY,
    supplier_id INTEGER REFERENCES suppliers(id),
    employee_id INTEGER REFERENCES employees(id),
    order_date DATE DEFAULT CURRENT_DATE,
    total_amount DECIMAL(12,2),
    status VARCHAR(20) DEFAULT 'pending',
    delivery_date DATE,
    notes TEXT
);

-- İndeksler (Performance için)
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_order_details_order ON order_details(order_id);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_supplier ON products(supplier_id);

-- Başarı mesajı
SELECT 'Database schema created successfully! 8 tables ready.' as result;