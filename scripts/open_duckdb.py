from pathlib import Path
import duckdb

project_root = Path(__file__).resolve().parents[1]
db_path = project_root / "data" / "warehouse" / "cultural_mood_tracker.duckdb"
db_path.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(db_path))

print("Connected to", db_path)
print(con.execute("SHOW TABLES").fetchall())