
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "features" / "web_features.csv"
OUTPUT_DIR = BASE_DIR / "data" / "features"


def prepare_features(df):
    """
    Model üçün uyğun xüsusiyyətləri hazırlayır.
    """

    df = df.copy()

    # Zaman xüsusiyyətlərini dövri formada kodlaşdırırıq.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # Rəqəmsal xüsusiyyətlər
    numeric_columns = [
        "path_length",
        "query_length",
        "special_char_count",
        "response_size",
        "query_present",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos"
    ]

    # Kateqorik xüsusiyyətlər
    categorical_columns = [
        "method",
        "user_agent_type"
    ]

    numeric_data = df[numeric_columns].astype(float)

    categorical_data = pd.get_dummies(
        df[categorical_columns],
        columns=categorical_columns,
        dtype=float
    )

    # Kateqorik və rəqəmsal məlumatları birləşdiririk.
    X = pd.concat(
        [numeric_data, categorical_data],
        axis=1
    )

    # Miqyas fərqlərini aradan qaldırırıq.
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled


def run_kmeans(X, df):
    """
    K-Means klasterləşdirmə alqoritmi.
    """

    print("\n========== K-MEANS ==========")

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X)

    df["kmeans_cluster"] = clusters

    print("\nCluster distribution:")
    print(df["kmeans_cluster"].value_counts().sort_index())

    score = silhouette_score(
        X,
        clusters,
        sample_size=min(3000, len(X)),
        random_state=42
    )

    print("\nSilhouette Score:", round(score, 4))

    return df


def run_dbscan(X, df):
    """
    DBSCAN klasterləşdirmə alqoritmi.

    Hesablama yükünü azaltmaq üçün
    5000 təsadüfi hadisə seçilir.
    """

    print("\n========== DBSCAN ==========")

    sample_size = min(5000, len(X))

    rng = np.random.default_rng(42)

    sample_indices = rng.choice(
        len(X),
        size=sample_size,
        replace=False
    )

    X_sample = X[sample_indices]

    # Ölçüləri azaldırıq.
    pca = PCA(
        n_components=8,
        random_state=42
    )

    X_reduced = pca.fit_transform(X_sample)

    model = DBSCAN(
        eps=2.5,
        min_samples=15
    )

    labels = model.fit_predict(X_reduced)

    dbscan_results = pd.DataFrame({
        "row_index": sample_indices,
        "dbscan_cluster": labels
    })

    print("\nDBSCAN cluster distribution:")
    print(
        dbscan_results["dbscan_cluster"]
        .value_counts()
        .sort_index()
    )

    noise_count = (labels == -1).sum()

    print("\nNoise events:", noise_count)

    print(
        "Noise percentage:",
        round(noise_count / sample_size * 100, 2),
        "%"
    )

    # Noise olmayan hadisələr üzrə silhouette
    non_noise_mask = labels != -1
    non_noise_labels = labels[non_noise_mask]
    non_noise_data = X_reduced[non_noise_mask]

    unique_clusters = np.unique(non_noise_labels)

    if (
        len(unique_clusters) >= 2
        and len(unique_clusters) < len(non_noise_labels)
    ):
        score = silhouette_score(
            non_noise_data,
            non_noise_labels,
            sample_size=min(2000, len(non_noise_labels)),
            random_state=42
        )

        print(
            "DBSCAN Silhouette Score:",
            round(score, 4)
        )
    else:
        print(
            "DBSCAN Silhouette Score hesablana bilmədi."
        )

    return dbscan_results


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Web feature dataset yüklənir...")

    df = pd.read_csv(INPUT_FILE)

    print("Dataset shape:", df.shape)

    # Model üçün giriş məlumatları
    X = prepare_features(df)

    print("Prepared feature shape:", X.shape)

    # K-Means
    df = run_kmeans(X, df)

    # DBSCAN
    dbscan_results = run_dbscan(X, df)

    # Nəticələri saxlayırıq.
    df.to_csv(
        OUTPUT_DIR / "web_kmeans_results.csv",
        index=False
    )

    dbscan_results.to_csv(
        OUTPUT_DIR / "web_dbscan_results.csv",
        index=False
    )

    print("\nNəticələr yadda saxlanıldı:")
    print("web_kmeans_results.csv")
    print("web_dbscan_results.csv")

    print("\nClustering mərhələsi tamamlandı.")


if __name__ == "__main__":
    main()
