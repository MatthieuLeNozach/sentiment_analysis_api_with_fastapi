-- Create the webapp database if it doesn't exist
SELECT 'CREATE DATABASE webapp' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'webapp')\gexec

-- Connect to the webapp database
\c webapp;

-- Create the users table if it doesn't exist
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(127) UNIQUE,
    first_name VARCHAR(127),
    last_name VARCHAR(127),
    country VARCHAR(127),
    hashed_password VARCHAR(255),
    is_active BOOLEAN DEFAULT FALSE,
    role VARCHAR(255),
    has_access_sentiment BOOLEAN DEFAULT FALSE,
    has_access_emotion BOOLEAN DEFAULT FALSE
);

-- Create the service_calls table if it doesn't exist
CREATE TABLE IF NOT EXISTS service_calls (
    id SERIAL PRIMARY KEY,
    service_version VARCHAR(2),
    success BOOLEAN DEFAULT FALSE,
    owner_id INTEGER REFERENCES users(id),
    request_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completion_time TIMESTAMP,
    duration FLOAT
);

-- Create the webapp_user and grant privileges
DO
$$
BEGIN
    IF NOT EXISTS (
        SELECT
        FROM   pg_catalog.pg_roles
        WHERE  rolname = 'webapp_user') THEN

        CREATE ROLE webapp_user LOGIN PASSWORD 'webapp_password';
    END IF;
END
$$;

GRANT ALL PRIVILEGES ON DATABASE webapp TO webapp_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO webapp_user;