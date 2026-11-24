
from pathlib import Path
from urllib.parse import unquote_plus

import pandas as pd

from src.preprocessing.preprocess import (
    clean_web_logs,
    clean_firewall_logs
)
from src.ingestion.loader import (
    load_web_logs,
    load_firewall_logs
)

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "data" / "features"


def create_web_features(df):
    df = clean_web_logs(df)

    # Yalnız etibarlı formatlı hadisələr üzərində işləyirik
    df = df[df["is_valid"]].copy()

    # URL-də kodlaşdırılmış simvolları normallaşdırırıq
    df["normalized_path"] = df["path"].fillna("").apply(
        lambda value: unquote_plus(str(value))
    )

    path = df["normalized_path"].str.lower()

    # SQLi üçün müdafiə yönümlü heuristic indikatorlar
    patterns = {
        "has_union_select": r"\bunion\b\s+\bselect\b",
        "has_drop_keyword": r"\bdrop\b",
        "has_sql_comment": r"--",
        "has_boolean_condition": r"\bor\b.*="
    }

    for feature_name, pattern in patterns.items():
        df[feature_name] = path.str.contains(
            pattern,
            regex=True,
            na=False
        )

    # Ümumi indikator sayı
    indicator_columns = list(patterns.keys())

    df["sqli_indicator_count"] = df[
        indicator_columns
    ].sum(axis=1)

    df["is_suspicious"] = (
        df["sqli_indicator_count"] > 0
    )

    # Path xüsusiyyətləri
    df["path_length"] = path.str.len()

    df["query_present"] = path.str.contains(
        r"\?", regex=True
    )

    df["query_length"] = path.apply(
        lambda value: len(value.split("?", 1)[1])
        if "?" in value else 0
    )

    df["special_char_count"] = path.str.count(
        r"[^a-zA-Z0-9]"
    )

    # HTTP status xüsusiyyətləri
    df["is_redirect"] = df["status"].isin([301, 302])
    df["is_client_error"] = df["status"].between(400, 499)
    df["is_server_error"] = df["status"].between(500, 599)
    df["is_error"] = df["status"].between(400, 599)

    # Zaman xüsusiyyətləri
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_of_week
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    # User-Agent kateqoriyası
    def classify_user_agent(value):
        value = str(value).lower()

        if "python-requests" in value:
            return "python_requests"
        elif "curl" in value:
            return "curl"
        elif "firefox" in value:
            return "firefox"
        elif "safari" in value:
            return "safari"
        elif "chrome" in value:
            return "chrome"
        else:
            return "other"

    df["user_agent_type"] = df["user_agent"].apply(
        classify_user_agent
    )

    # Cavab ölçüsü
    df["response_size"] = df["bytes"]

    return df


def create_firewall_features(df):
    df = clean_firewall_logs(df)

    df = df[df["is_valid"]].copy()

    # Ümumi trafik həcmi
    df["total_bytes"] = (
        df["bytes_sent"] + df["bytes_received"]
    )

    # Göndərilən və qəbul edilən trafik nisbəti
    df["bytes_ratio"] = (
        df["bytes_sent"] /
        (df["bytes_received"] + 1)
    )

    # Port diapazonu
    df["dst_port_group"] = pd.cut(
        df["dst_port"],
        bins=[0, 1023, 49151, 65535],
        labels=["well_known", "registered", "dynamic"]
    )

    # Zaman xüsusiyyətləri
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_of_week

    # Firewall nəticəsi
    df["is_allowed"] = df["action"].eq("ALLOW")
    df["is_denied"] = df["action"].eq("DENY")

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # İlkin məlumatları yükləyirik
    web_raw = load_web_logs()
    firewall_raw = load_firewall_logs()

    # Feature Engineering
    web_features = create_web_features(web_raw)
    firewall_features = create_firewall_features(firewall_raw)

    # Nəticələri saxlayırıq
    web_features.to_csv(
        OUTPUT_DIR / "web_features.csv",
        index=False
    )

    firewall_features.to_csv(
        OUTPUT_DIR / "firewall_features.csv",
        index=False
    )

    # Web nəticələrinin xülasəsi
    print("\nWEB FEATURE SUMMARY")
    print("Rows:", len(web_features))
    print(
        "Suspicious requests:",
        web_features["is_suspicious"].sum()
    )

    print("\nSQLi indicator counts:")
    print(
        web_features[
            [
                "has_union_select",
                "has_drop_keyword",
                "has_sql_comment",
                "has_boolean_condition"
            ]
        ].sum()
    )

    print("\nSuspicious requests by HTTP status:")
    print(
        web_features[
            web_features["is_suspicious"]
        ]["status"].value_counts().sort_index()
    )

    print("\nUser-Agent distribution:")
    print(web_features["user_agent_type"].value_counts())

    # Firewall nəticələrinin xülasəsi
    print("\nFIREWALL FEATURE SUMMARY")
    print("Rows:", len(firewall_features))
    print(
        "Total traffic bytes:",
        firewall_features["total_bytes"].sum()
    )
    print(
        "Allowed events:",
        firewall_features["is_allowed"].sum()
    )
    print(
        "Denied events:",
        firewall_features["is_denied"].sum()
    )

    print("\nFeature Engineering completed.")


if __name__ == "__main__":
    main()
