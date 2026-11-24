
import pandas as pd

from src.ingestion.loader import (
    load_web_logs,
    load_firewall_logs
)


def inspect_dataset(df, name):
    print("\n" + "=" * 60)
    print(f"{name} - DATASET OVERVIEW")
    print("=" * 60)

    # Dataset ölçüsü
    print("\n[1] Dataset shape:")
    print(df.shape)

    # Sütunların məlumat tipləri
    print("\n[2] Data types:")
    print(df.dtypes)

    # Boş dəyərlər
    print("\n[3] Missing values:")
    print(df.isnull().sum())

    # Təkrarlanan sətirlər
    print("\n[4] Duplicate rows:")
    print(df.duplicated().sum())

    # Statistik məlumat
    print("\n[5] Statistical summary:")
    print(df.describe(include="all").T)

    # Tarix aralığı
    print("\n[6] Timestamp range:")
    if "timestamp" in df.columns:
        print("Start:", df["timestamp"].min())
        print("End:  ", df["timestamp"].max())


def analyze_web_logs(df):
    print("\n" + "=" * 60)
    print("WEB LOG ANALYSIS")
    print("=" * 60)

    print("\n[1] HTTP Methods:")
    print(df["method"].value_counts())

    print("\n[2] HTTP Status Codes:")
    print(df["status"].value_counts().sort_index())

    print("\n[3] User Agents:")
    print(df["user_agent"].value_counts())

    # SQL Injection üçün ilkin indikatorlar.
    # Bunlar şübhəli nümunələri işarələyir, hücumun uğurunu sübut etmir.
    path = df["path"].fillna("").astype(str)

    patterns = {
        "union_select": r"\bUNION\b.*\bSELECT\b",
        "drop_keyword": r"\bDROP\b",
        "sql_comment": r"--",
        "boolean_condition": r"\bOR\b.*="
    }

    print("\n[4] SQLi heuristic indicators:")

    for name, pattern in patterns.items():
        count = path.str.contains(
            pattern,
            case=False,
            regex=True
        ).sum()

        print(f"{name}: {count}")

    # Hər hadisəyə ayrıca indikator sütunları əlavə edirik.
    for name, pattern in patterns.items():
        df[name] = path.str.contains(
            pattern,
            case=False,
            regex=True
        )

    df["suspicious_indicator_count"] = df[
        list(patterns.keys())
    ].sum(axis=1)

    print("\n[5] Requests with at least one indicator:")
    print(
        (df["suspicious_indicator_count"] > 0).sum()
    )

    print("\n[6] Status codes for flagged requests:")
    flagged = df[df["suspicious_indicator_count"] > 0]

    print(flagged["status"].value_counts().sort_index())

    return df


def analyze_firewall_logs(df):
    print("\n" + "=" * 60)
    print("FIREWALL ANALYSIS")
    print("=" * 60)

    print("\n[1] Firewall actions:")
    print(df["action"].value_counts())

    print("\n[2] Protocol distribution:")
    print(df["protocol"].value_counts())

    print("\n[3] Destination ports:")
    print(df["dst_port"].value_counts().sort_index())

    print("\n[4] Destination countries:")
    print(df["dst_country"].value_counts())


def main():
    web_df = load_web_logs()
    firewall_df = load_firewall_logs()

    inspect_dataset(web_df, "WEB ACCESS")
    inspect_dataset(firewall_df, "FIREWALL")

    web_df = analyze_web_logs(web_df)
    analyze_firewall_logs(firewall_df)

    print("\nExploration completed.")


if __name__ == "__main__":
    main()
