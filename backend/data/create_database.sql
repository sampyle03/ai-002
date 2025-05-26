\c postgres;
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'traintalk_admin') THEN
        CREATE ROLE traintalk_admin WITH LOGIN PASSWORD 'traintalk_password123';
    END IF;
END $$;

SELECT 'CREATE DATABASE traintalk'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'footballblog_db')\gexec





\c footballblog_db; 

https://www.youtube.com/watch?v=fjH-3MNY94k&pp=ygUGI21hd3Ro