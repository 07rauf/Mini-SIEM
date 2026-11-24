
from pathlib import Path
import pandas as pd

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Raw dataset directory
DATA_DIR = BASE_DIR / "data" / "raw" / "P07"


def load_web_logs():
    file_path = DATA_DIR / "web_access.csv"

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    return df


def load_firewall_logs():
    file_path = DATA_DIR / "firewall.csv"

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    return df


if __name__ == "__main__":

    web_df = load_web_logs()
    firewall_df = load_firewall_logs()

    print("WEB LOGS")
    print(web_df.shape)
    print(web_df.head())

    print("\nFIREWALL LOGS")
    print(firewall_df.shape)
    print(firewall_df.head())
