import sqlite3
import json
from typing import List, Optional

DB_PATH = "./university_users.db"

def init_db():
    """
    Create users table if not exists
    Stores user_id, password, role, and folder access list
    """
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id     TEXT PRIMARY KEY,
            password    TEXT NOT NULL,
            role        TEXT NOT NULL,
            department  TEXT NOT NULL,
            access_dirs TEXT NOT NULL
        )
    """)

    # Insert sample users if table is empty
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        sample_users = [
            # (user_id, password, role, department, access_dirs as JSON)
            ("cse_student",  "pass123", "student",  "CSE",
             json.dumps(["academic/CSE"])),

            ("ece_student",  "pass123", "student",  "ECE",
             json.dumps(["academic/ECE"])),

            ("mech_student", "pass123", "student",  "Mechanical",
             json.dumps(["academic/Mechanical"])),

            ("civil_student","pass123", "student",  "Civil",
             json.dumps(["academic/Civil"])),

            ("mba_student",  "pass123", "student",  "MBA",
             json.dumps(["academic/MBA"])),

            ("admin_user",   "admin123","admin",    "Administration",
             json.dumps(["administration"])),

            ("hod_cse",      "hod123",  "hod",      "CSE",
             json.dumps(["academic/CSE", "administration"])),

            ("principal",    "prin123", "principal","ALL",
             json.dumps([
                 "academic/CSE",
                 "academic/ECE",
                 "academic/Mechanical",
                 "academic/Civil",
                 "academic/MBA",
                 "administration"
             ])),
        ]

        c.executemany(
            "INSERT INTO users VALUES (?,?,?,?,?)",
            sample_users
        )
        conn.commit()
        print("Sample users created in DB")

    conn.close()


def get_user(user_id: str, password: str) -> Optional[dict]:
    """
    Authenticate user and return their details
    Returns None if user not found or wrong password
    """
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute(
        "SELECT * FROM users WHERE user_id=? AND password=?",
        (user_id, password)
    )
    row = c.fetchone()
    conn.close()

    if row:
        return {
            "user_id"    : row[0],
            "role"       : row[2],
            "department" : row[3],
            "access_dirs": json.loads(row[4])  # parse JSON list
        }
    return None


def get_all_users() -> List[dict]:
    """Get all users for admin view"""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT user_id, role, department, access_dirs FROM users")
    rows = conn.fetchall()
    conn.close()

    return [
        {
            "user_id"    : r[0],
            "role"       : r[1],
            "department" : r[2],
            "access_dirs": json.loads(r[3])
        }
        for r in rows
    ]


def add_user(
    user_id    : str,
    password   : str,
    role       : str,
    department : str,
    access_dirs: List[str]
):
    """Add new user to DB"""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    try:
        c.execute(
            "INSERT INTO users VALUES (?,?,?,?,?)",
            (user_id, password, role, department, json.dumps(access_dirs))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # user already exists
    finally:
        conn.close()
