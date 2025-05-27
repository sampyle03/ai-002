import psycopg2
import csv
import os

def insert_stations(csv_path):
    # 1) connect to your database
    conn = psycopg2.connect(
        host="localhost",
        database="traintalk",
        user="postgres",
        password="password"
    )
    cur = conn.cursor()

    # 2) open & read the CSV
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # pick alias if present, else the name
            alias = row['longname.name_alias'].strip()
            station_name = alias if alias and alias != r'\N' else row['name'].strip()

            # use the tiploc code as our station_code
            station_code = row['tiploc'].strip()
            if not station_code or station_code == r'\N':
                # skip rows without a valid code
                continue

            try:
                cur.execute(
                    """
                    INSERT INTO stations (station_name, station_code)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (station_name, station_code)
                )
            except Exception as e:
                print(f"Failed to insert {station_name} ({station_code}): {e}")

    # 3) commit + cleanup
    conn.commit()
    cur.close()
    conn.close()
    print("Stations insertion complete.")

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    insert_stations(os.path.join(here, 'stations.csv'))
