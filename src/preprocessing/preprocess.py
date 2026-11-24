
from pathlib import Path
import ipaddress
import pandas as pd

from src.ingestion.loader import (
    load_web_logs,
    load_firewall_logs
)

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "data" / "processed"


def is_valid_ip(value):
    try:
        ipaddress.ip_address(str(value))
        return True
    except ValueError:
        return False


def clean_web_logs(df):
    df = df.copy()

    # Mətn sütunlarını standartlaşdırırıq
    for col in ["src_ip", "method", "path", "user_agent", "referrer"]:
        df[col] = df[col].astype("string").str.strip()

    df["method"] = df["method"].str.upper()

    # Tarix və rəqəmsal sütunlar
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )

    df["status"] = pd.to_numeric(
        df["status"], errors="coerce"
    )

    df["bytes"] = pd.to_numeric(
        df["bytes"], errors="coerce"
    )

    # Keyfiyyət yoxlamaları
    df["valid_ip"] = df["src_ip"].apply(is_valid_ip)
    df["valid_timestamp"] = df["timestamp"].notna()
    df["valid_status"] = df["status"].between(100, 599)
    df["valid_bytes"] = df["bytes"].ge(0)

    # Bütün yoxlamalardan keçən sətirlər
    validation_cols = [
        "valid_ip",
        "valid_timestamp",
        "valid_status",
        "valid_bytes"
    ]

    df["is_valid"] = df[validation_cols].all(axis=1)

    return df


def clean_firewall_logs(df):
    df = df.copy()

    # Mətn sütunlarını standartlaşdırırıq
    text_columns = [
        "src_ip", "dst_ip", "protocol",
        "action", "dst_country"
    ]

    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()

    df["protocol"] = df["protocol"].str.upper()
    df["action"] = df["action"].str.upper()
    df["dst_country"] = df["dst_country"].str.upper()

    # Tarix və rəqəmsal sütunlar
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )

    numeric_columns = [
        "src_port", "dst_port",
        "bytes_sent", "bytes_received"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Keyfiyyət yoxlamaları
    df["valid_src_ip"] = df["src_ip"].apply(is_valid_ip)
    df["valid_dst_ip"] = df["dst_ip"].apply(is_valid_ip)
    df["valid_timestamp"] = df["timestamp"].notna()

    df["valid_ports"] = (
        df["src_port"].between(1, 65535)
        & df["dst_port"].between(1, 65535)
    )

    df["valid_bytes"] = (
        df["bytes_sent"].ge(0)
        & df["bytes_received"].ge(0)
    )

    validation_cols = [
        "valid_src_ip",
        "valid_dst_ip",
        "valid_timestamp",
        "valid_ports",
        "valid_bytes"
    ]

    df["is_valid"] = df[validation_cols].all(axis=1)

    return df


def print_validation_report(df, name):
    print("\n" + "=" * 50)
    print(f"{name} VALIDATION REPORT")
    print("=" * 50)

    print("Total rows:", len(df))
    print("Valid rows:", df["is_valid"].sum())
    print("Invalid rows:", (~df["is_valid"]).sum())

    print("\nValidation details:")

    validation_columns = [
        col for col in df.columns
        if col.startswith("valid_")
    ]

    for col in validation_columns:
        print(f"{col}: {df[col].sum()} / {len(df)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Raw logları yükləyirik
    web_raw = load_web_logs()
    firewall_raw = load_firewall_logs()

    # Təmizləmə və yoxlama
    web_clean = clean_web_logs(web_raw)
    firewall_clean = clean_firewall_logs(firewall_raw)

    # Nəticələri göstəririk
    print_validation_report(web_clean, "WEB LOGS")
    print_validation_report(firewall_clean, "FIREWALL LOGS")

    # Bütün sətirləri saxlayırıq.
    # Etibarsız sətirlər də sonrakı araşdırma üçün qorunur.
    web_clean.to_csv(
        OUTPUT_DIR / "web_access_clean.csv",
        index=False
    )

    firewall_clean.to_csv(
        OUTPUT_DIR / "firewall_clean.csv",
        index=False
    )

    # Etibarsız sətirləri ayrıca saxlayırıq
    web_clean[~web_clean["is_valid"]].to_csv(
        OUTPUT_DIR / "web_access_invalid.csv",
        index=False
    )

    firewall_clean[~firewall_clean["is_valid"]].to_csv(
        OUTPUT_DIR / "firewall_invalid.csv",
        index=False
    )

    print("\nPreprocessing completed.")
    print("Output directory:", OUTPUT_DIR)

    
if __name__ == "__main__":
    main()
