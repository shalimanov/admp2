import os
import time
import csv
from datetime import datetime
from collections import Counter

from src.apriori import apriori, generate_rules
from src.fpgrowth import fpgrowth
from src.clustering import KMeans, KMedians, AgglomerativeSingleLink, DBSCAN

# Для дендрограми
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT, "data")
REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_groceries():
    path = os.path.join(DATA_DIR, "groceries.csv")
    transactions = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            transactions.append(set(row))
    return transactions


def load_mall():
    path = os.path.join(DATA_DIR, "Mall_Customers.csv")
    points = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append((
                float(row['Annual Income (k$)']),
                float(row['Spending Score (1-100)'])
            ))
    return points


def rule_mining(transactions):
    n_trans = len(transactions)
    # Apriori
    freq = apriori(transactions, 0.02)
    rules = generate_rules(freq, 0.3)
    file_apriori = os.path.join(REPORT_DIR, "apriori_report.txt")
    with open(file_apriori, "w") as f:
        f.write(f"Apriori Report  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"Transactions: {n_trans}\n")
        f.write(f"Frequent itemsets: {len(freq)}\n")
        f.write(f"Rules generated: {len(rules)}\n\n")
        f.write("Top-10 rules by confidence:\n")
        top_rules = sorted(rules, key=lambda x: x[2], reverse=True)[:10]
        for antecedent, consequent, conf, support, lift in top_rules:
            f.write(
                f"{set(antecedent)} -> {set(consequent)} "
                f"sup={support:.3f} conf={conf:.3f} lift={lift:.2f}\n"
            )
    # FP-Growth
    freq_fp = fpgrowth(transactions, 0.02)
    file_fpg = os.path.join(REPORT_DIR, "fpgrowth_report.txt")
    with open(file_fpg, "w") as f:
        f.write(f"FP-Growth Report  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"Transactions: {n_trans}\n")
        f.write(f"Frequent itemsets: {len(freq_fp)}\n")
    return file_apriori, file_fpg


def clustering_analysis(points):
    file_clust = os.path.join(REPORT_DIR, "clustering_report.txt")
    with open(file_clust, "w") as f:
        f.write(f"Clustering Report  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"Samples: {len(points)}\n\n")
        f.write("k, KMeans inertia, KMedians total L1\n")
        for k in range(2, 9):
            km = KMeans(k, random_state=0).fit(points)
            med = KMedians(k, random_state=0).fit(points)
            labels_med = med.predict(points)
            total_l1 = sum(
                sum(abs(x - y) for x, y in zip(points[i], med.centroids[labels_med[i]]))
                for i in range(len(points))
            )
            f.write(f"{k}, {km.inertia_:.2f}, {total_l1:.2f}\n")
        # Ієрархічний single-link
        ag = AgglomerativeSingleLink(5).fit(points)
        labels_h = ag.labels_
        n_h_clusters = len(set(labels_h))
        f.write(f"\nSingle-link hierarchical — k=5, clusters formed: {n_h_clusters}\n")
        sizes = Counter(labels_h)
        f.write(f"  Cluster sizes: {list(sizes.values())}\n")
        # DBSCAN
        db = DBSCAN(eps=15, min_samples=5).fit(points)
        labels_db = db.labels_
        n_db_clusters = len({lbl for lbl in labels_db if lbl != -1})
        noise = sum(1 for lbl in labels_db if lbl == -1)
        f.write(f"DBSCAN eps=15, min_samples=5 — clusters: {n_db_clusters}, noise points: {noise}\n")
        main_size = len(points) - noise
        noise_idxs = [i for i, lbl in enumerate(labels_db) if lbl == -1]
        f.write(f"  Main cluster size: {main_size}\n")
        f.write(f"  Noise indices: {noise_idxs}\n")
    return file_clust


def plot_dendrogram(points, output_path):
    """
    Створює дендрограму single-link кластеризації та зберігає у файл.
    """
    Z = linkage(points, method='single', metric='euclidean')
    plt.figure(figsize=(10, 5))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=8, no_labels=True)
    plt.title("Single-Linkage Dendrogram")
    plt.xlabel("Індекс зразка")
    plt.ylabel("Відстань")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    start = time.time()

    # Ассоціативні правила
    transactions = load_groceries()
    file_apriori, file_fpg = rule_mining(transactions)

    # Кластеризація
    points = load_mall()
    file_clust = clustering_analysis(points)

    # Дендрограма
    dendro_path = os.path.join(REPORT_DIR, "dendrogram.png")
    plot_dendrogram(points, dendro_path)

    elapsed = time.time() - start
    print("Reports generated:")
    print(f"  {file_apriori}")
    print(f"  {file_fpg}")
    print(f"  {file_clust}")
    print(f"  {dendro_path}")
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
