from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data(path):
    return pd.read_csv(path)


def save_processed_data(df,filename):
    df.to_csv(PROCESSED_DATA_DIR / filename, index=False)
    return PROCESSED_DATA_DIR


def load_processed_data():
    return pd.read_csv(PROCESSED_DATA_DIR / "processed_students.csv")

