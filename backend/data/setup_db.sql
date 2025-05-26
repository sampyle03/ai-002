
DROP TABLE IF EXISTS posts;
DROP TABLE IF EXISTS users;
DROP FUNCTION IF EXISTS update_updated_at_column();

CREATE TABLE stations (
    id SERIAL PRIMARY KEY,
    station_name VARCHAR(50) UNIQUE NOT NULL,
    station_code VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE railcards (
    id SERIAL PRIMARY KEY,
    railcard VARCHAR(50) UNIQUE NOT NULL,
    railcard_code VARCHAR(3) UNIQUE NOT NULL
);


-- Create a function that will automatically update the updated_at timestamp
-- This function will be called by the trigger below
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    -- Set the updated_at column to the current timestamp
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a trigger that automatically updates the updated_at timestamp
-- This trigger fires BEFORE any UPDATE operation on the posts table
-- It ensures that updated_at is always current when a post is modified
CREATE TRIGGER update_posts_updated_at
    BEFORE UPDATE ON posts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 

-- Grant all permissions in the database to blog_user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO blog_user;
GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO blog_user;

-- To add column user_secret to users table without losing data, use the following command: --
-- ALTER TABLE users ADD COLUMN user_secret VARCHAR(255); --
-- ALTER TABLE users ADD COLUMN graphical_password VARCHAR(255); --