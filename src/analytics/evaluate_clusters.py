
from pathlib import Path
import pandas as pd


# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_DIR = BASE_DIR / "data" / "features"


# =====================================================
# 1. K-MEANS EVALUATION
# =====================================================

def evaluate_kmeans():

    print("\n========== K-MEANS EVALUATION ==========")

    file_path = FEATURE_DIR / "web_kmeans_results.csv"
    df = pd.read_csv(file_path)

    # Klasterlər və şübhəli sorğular arasındakı əlaqə
    table = pd.crosstab(
        df["kmeans_cluster"],
        df["is_suspicious"],
        margins=True
    )

    print("\nCluster vs suspicious:")
    print(table)

    # Klasterlər üzrə ümumi və şübhəli sorğular
    summary = df.groupby("kmeans_cluster").agg(
        total_requests=("is_suspicious", "size"),
        suspicious_requests=("is_suspicious", "sum")
    )

    summary["normal_requests"] = (
        summary["total_requests"]
        - summary["suspicious_requests"]
    )

    summary["suspicious_percentage"] = (
        summary["suspicious_requests"]
        / summary["total_requests"] * 100
    ).round(2)

    print("\nK-Means suspicious percentage:")
    print(summary)

    # Nəticələri CSV formatında saxlayırıq
    summary.to_csv(
        FEATURE_DIR / "kmeans_evaluation.csv"
    )

    return df, summary


# =====================================================
# 2. DBSCAN EVALUATION
# =====================================================

def evaluate_dbscan():

    print("\n========== DBSCAN EVALUATION ==========")

    features = pd.read_csv(
        FEATURE_DIR / "web_features.csv"
    )

    results = pd.read_csv(
        FEATURE_DIR / "web_dbscan_results.csv"
    )

    # DBSCAN nəticələrindəki indeksləri
    # ilkin feature dataset ilə uyğunlaşdırırıq.
    labels = features[
        ["is_suspicious"]
    ].reset_index()

    labels = labels.rename(
        columns={"index": "row_index"}
    )

    merged = results.merge(
        labels,
        on="row_index",
        how="left",
        validate="one_to_one"
    )

    # Klaster və şübhəli sorğuların müqayisəsi
    table = pd.crosstab(
        merged["dbscan_cluster"],
        merged["is_suspicious"],
        margins=True
    )

    print("\nDBSCAN cluster vs suspicious:")
    print(table)

    # Klasterlər üzrə statistik göstəricilər
    summary = merged.groupby("dbscan_cluster").agg(
        total_requests=("is_suspicious", "size"),
        suspicious_requests=("is_suspicious", "sum")
    )

    summary["normal_requests"] = (
        summary["total_requests"]
        - summary["suspicious_requests"]
    )

    summary["suspicious_percentage"] = (
        summary["suspicious_requests"]
        / summary["total_requests"] * 100
    ).round(2)

    print("\nDBSCAN suspicious percentage:")
    print(summary)

    # Noise statistikası
    noise_count = (
        merged["dbscan_cluster"] == -1
    ).sum()

    noise_percentage = (
        noise_count / len(merged) * 100
    )

    print("\nDBSCAN noise events:", noise_count)
    print(
        "DBSCAN noise percentage:",
        round(noise_percentage, 2),
        "%"
    )

    print(
        "\nDBSCAN sample suspicious events:",
        merged["is_suspicious"].sum()
    )

    # Nəticələri saxlayırıq
    summary.to_csv(
        FEATURE_DIR / "dbscan_evaluation.csv"
    )

    return merged, summary


# =====================================================
# 3. K-MEANS CLUSTER PROFILES
# =====================================================

def analyze_cluster_profiles():

    print("\n========== CLUSTER PROFILES ==========")

    df = pd.read_csv(
        FEATURE_DIR / "web_kmeans_results.csv"
    )

    # Rəqəmsal xüsusiyyətlər
    numeric_features = [
        "path_length",
        "query_length",
        "special_char_count",
        "response_size",
        "hour"
    ]

    # Klasterlər üzrə orta göstəricilər
    profile = df.groupby("kmeans_cluster")[
        numeric_features
    ].mean().round(2)

    print("\nNumeric feature averages:")
    print(profile)

    # HTTP metodlarının bölgüsü
    method_distribution = pd.crosstab(
        df["kmeans_cluster"],
        df["method"]
    )

    print("\nHTTP method distribution:")
    print(method_distribution)

    # User-Agent bölgüsü
    user_agent_distribution = pd.crosstab(
        df["kmeans_cluster"],
        df["user_agent_type"]
    )

    print("\nUser-Agent distribution:")
    print(user_agent_distribution)

    # Hər klasterdə HTTP status kodları
    status_distribution = pd.crosstab(
        df["kmeans_cluster"],
        df["status"]
    )

    print("\nHTTP status distribution:")
    print(status_distribution)

    # Nəticələri ayrıca fayllarda saxlayırıq
    profile.to_csv(
        FEATURE_DIR / "cluster_profiles.csv"
    )

    method_distribution.to_csv(
        FEATURE_DIR / "cluster_method_distribution.csv"
    )

    user_agent_distribution.to_csv(
        FEATURE_DIR / "cluster_user_agent_distribution.csv"
    )

    status_distribution.to_csv(
        FEATURE_DIR / "cluster_status_distribution.csv"
    )

    return profile


# =====================================================
# 4. FINAL SUMMARY
# =====================================================

def print_final_summary(kmeans_summary, dbscan_summary):

    print("\n========== FINAL CLUSTERING SUMMARY ==========")

    print(
        "\nTotal K-Means clusters:",
        len(kmeans_summary)
    )

    print(
        "K-Means suspicious requests:",
        kmeans_summary["suspicious_requests"].sum()
    )

    print(
        "\nTotal DBSCAN clusters (excluding noise):",
        len([
            cluster
            for cluster in dbscan_summary.index
            if cluster != -1
        ])
    )

    if -1 in dbscan_summary.index:

        print(
            "DBSCAN noise events:",
            dbscan_summary.loc[-1, "total_requests"]
        )

    print("\nAll evaluation files have been generated.")


# =====================================================
# MAIN
# =====================================================

def main():

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # K-Means analizi
    kmeans_df, kmeans_summary = evaluate_kmeans()

    # DBSCAN analizi
    dbscan_df, dbscan_summary = evaluate_dbscan()

    # Klaster profilləri
    analyze_cluster_profiles()

    # Yekun statistik xülasə
    print_final_summary(
        kmeans_summary,
        dbscan_summary
    )

    print("\nCluster evaluation completed successfully.")


if __name__ == "__main__":
    main()
