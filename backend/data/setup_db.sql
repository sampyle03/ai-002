
DROP TABLE IF EXISTS stations;
DROP TABLE IF EXISTS railcards;

CREATE TABLE stations (
    id SERIAL PRIMARY KEY,
    station_name VARCHAR(255) UNIQUE NOT NULL,
    station_code VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE railcards (
    id SERIAL PRIMARY KEY,
    railcard VARCHAR(50) UNIQUE NOT NULL,
    railcard_code VARCHAR(3) UNIQUE NOT NULL
);

CREATE TABLE chats (
    id SERIAL PRIMARY KEY,
    messages JSONB NOT NULL,
    current_slots JSONB NOT NULL,
    required_slots JSONB NOT NULL,
    current_slots_delay JSONB NOT NULL,
    current_task VARCHAR(50),
    temp VARCHAR(50)
    );

