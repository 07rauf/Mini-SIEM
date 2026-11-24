
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from src.clustering.cluster_web import prepare_features


# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_DIR = BASE_DIR / "data" / "features"
FIGURE_DIR = BASE_DIR / "reports" / "figures"


# =====================================================
# 1. K-MEANS VISUALIZATION
# =====================================================

def visualize_kmeans():

    print("\n========== K-MEANS VISUALIZATION ==========")

    df = pd.read_csv(
        FEATURE_DIR / "web_kmeans_results.csv"
    )

    # Eyni xüsusiyyətləri yenidən hazırlayırıq
    X = prepare_features(df)

    # 2 ölçülü PCA
    pca = PCA(n_components=2, random_state=42)

    X_pca = pca.fit_transform(X)

    # PCA nəticələri
    print(
        "Explained variance ratio:",
        pca.explained_variance_ratio_.round(4)
    )

    print(
        "Total explained variance:",
        round(pca.explained_variance_ratio_.sum(), 4)
    )

    plt.figure(figsize=(11, 7))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=df["kmeans_cluster"],
        cmap="tab10",
        s=15,
        alpha=0.6
    )

    plt.title("K-Means Clustering - Web Access Logs")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.colorbar(
        scatter,
        label="Cluster"
    )

    plt.grid(alpha=0.2)
    plt.tight_layout()

    output_file = FIGURE_DIR / "kmeans_clusters.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("K-Means visualization saved:", output_file)


# =====================================================
# 2. DBSCAN VISUALIZATION
# =====================================================

def visualize_dbscan():

    print("\n========== DBSCAN VISUALIZATION ==========")

    df = pd.read_csv(
        FEATURE_DIR / "web_features.csv"
    )

    results = pd.read_csv(
        FEATURE_DIR / "web_dbscan_results.csv"
    )

    # Eyni giriş xüsusiyyətləri
    X = prepare_features(df)

    # DBSCAN üçün seçilmiş sətirlərin indeksləri
    sample_indices = results["row_index"].to_numpy()

    X_sample = X[sample_indices]

    # 2 ölçülü PCA - yalnız vizuallaşdırma
    pca = PCA(n_components=2, random_state=42)

    X_pca = pca.fit_transform(X_sample)

    labels = results["dbscan_cluster"].to_numpy()

    print(
        "Explained variance ratio:",
        pca.explained_variance_ratio_.round(4)
    )

    print(
        "Total explained variance:",
        round(pca.explained_variance_ratio_.sum(), 4)
    )

    plt.figure(figsize=(11, 7))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="tab10",
        s=15,
        alpha=0.6
    )

    plt.title("DBSCAN Clustering - Web Access Logs")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.colorbar(
        scatter,
        label="Cluster (-1 = Noise)"
    )

    plt.grid(alpha=0.2)
    plt.tight_layout()

    output_file = FIGURE_DIR / "dbscan_clusters.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("DBSCAN visualization saved:", output_file)


# =====================================================
# MAIN
# =====================================================

def main():

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    visualize_kmeans()
    visualize_dbscan()

    print("\nAll clustering visualizations completed.")


if __name__ == "__main__":
    main()
