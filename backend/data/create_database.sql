\c postgres;
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'traintalk_admin') THEN
        CREATE ROLE traintalk_admin WITH LOGIN PASSWORD 'traintalk_password123';
    END IF;
END $$;

SELECT 'CREATE DATABASE traintalk'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'traintalk_db')\gexec





\c traintalk_db; 
