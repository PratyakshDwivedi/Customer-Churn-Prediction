-- Example seed rows for the `customers` table.
--
-- The *full* dataset (several thousand rows) is synthetic and generated
-- reproducibly by `scripts/setup_database.py`, which bulk-inserts it via
-- pandas. These few rows document the exact column order and value format
-- (note total_charges as text and churn as 'Yes'/'No') and let you seed a
-- minimal database by hand with:  sqlite3 database/churn.db < database/seed.sql

INSERT INTO customers (
    customer_id, age, gender, senior_citizen, partner, dependents,
    tenure_months, contract_type, payment_method, internet_service,
    tech_support, online_security, paperless_billing,
    monthly_charges, total_charges, churn
) VALUES
    ('C000001', 34, 'Female', 0, 'Yes', 'No', 24, 'One year', 'Credit card',
     'DSL', 'Yes', 'No', 'Yes', 58.30, '1399.20', 'No'),
    ('C000002', 71, 'Male', 1, 'No', 'No', 3, 'Month-to-month', 'Electronic check',
     'Fiber optic', 'No', 'No', 'Yes', 89.10, '267.30', 'Yes'),
    ('C000003', 45, 'Male', 0, 'Yes', 'Yes', 60, 'Two year', 'Bank transfer',
     'DSL', 'Yes', 'Yes', 'No', 52.75, '3165.00', 'No');
