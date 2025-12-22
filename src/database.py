import sqlite3
import pandas as pd
from pathlib import Path


DB_PATH = Path(__file__).parent.parent / "data" / "processed" / "credit.db"


def setup_db(csv_path: str, db_path: str = str(DB_PATH)) -> None:
    """Load raw CSV into a SQLite database as the 'borrowers' table."""
    df = pd.read_csv(csv_path, index_col=0)
    df = df.reset_index(drop=True)
    df.index.name = "borrower_id"

    con = sqlite3.connect(db_path)
    df.to_sql("borrowers", con, if_exists="replace", index=True)
    con.close()
    print(f"Database created at {db_path}  ({len(df):,} rows)")


def query(sql: str, db_path: str = str(DB_PATH)) -> pd.DataFrame:
    """Run a SQL query against the SQLite database and return a DataFrame."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(sql, con)
    con.close()
    return df


def get_connection(db_path: str = str(DB_PATH)) -> sqlite3.Connection:
    return sqlite3.connect(db_path)
