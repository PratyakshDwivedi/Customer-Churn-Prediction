-- Schema for the customer churn database (SQLite).
--
-- The raw feed mirrors a production export: numeric columns that can
-- arrive dirty (total_charges as text, nullable fields) are typed loosely
-- on purpose so the Python validation/cleaning layer stays responsible
-- for data quality rather than the database rejecting rows silently.

DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id        TEXT,            -- not UNIQUE on purpose: duplicates must be caught in code
    age                INTEGER,
    gender             TEXT,
    senior_citizen     INTEGER,
    partner            TEXT,
    dependents         TEXT,
    tenure_months      INTEGER,
    contract_type      TEXT,
    payment_method     TEXT,
    internet_service   TEXT,
    tech_support       TEXT,
    online_security    TEXT,
    paperless_billing  TEXT,
    monthly_charges    REAL,
    total_charges      TEXT,            -- stored as text to mimic the real Telco export
    churn              TEXT             -- 'Yes' / 'No'
);

-- An index on customer_id speeds up lookups without enforcing uniqueness.
CREATE INDEX IF NOT EXISTS idx_customers_customer_id ON customers (customer_id);
