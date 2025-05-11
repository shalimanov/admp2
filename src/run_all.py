
"""Run all LAB2 algorithms and generate console output + simple text report.

Usage:
    python -m src.run_all
"""
import os, csv, urllib.request, time, math, random, statistics
from datetime import datetime
from .apriori import apriori, generate_rules
from .fpgrowth import fpgrowth
from .clustering import KMeans, KMedians, AgglomerativeSingleLink, DBSCAN

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT, "data")
REPORT_PATH = os.path.join(ROOT, "report.txt")

# URLs to raw CSV mirrors (no auth required)
DATASETS = {
    "groceries.csv": "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv",
    "Mall_Customers.csv": "https://raw.githubusercontent.com/SawyerRen/mall-customer-segmentation-data/master/Mall_Customers.csv"
}

def download_if_missing():
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname, url in DATASETS.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"[INFO] {fname} not found — downloading…")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"[OK]  Saved to data/{fname}")
            except Exception as e:
                print(f"[WARN] Could not download {fname}: {e}")
                print(f"       Please place the file manually into data/ and rerun.")
                continue

def load_groceries():
    path = os.path.join(DATA_DIR, "groceries.csv")
    transactions = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                transactions.append(set(item.strip() for item in row if item.strip()))
    return transactions

def load_mall():
    path = os.path.join(DATA_DIR, "Mall_Customers.csv")
    points = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append((
                    float(row["Annual Income (k$)"]),
                    float(row["Spending Score (1-100)"])
                ))
            except KeyError:
                # fallback for slightly different column names
                points.append((
                    float(row.get("Annual Income (k$)", row.get("Annual_Income_(k$)"))),
                    float(row.get("Spending Score (1-100)", row.get("Spending_Score")))
                ))
    return points

def rule_mining(transactions, report_lines):
    report_lines.append("\n=== Association Rule Mining ===")
    n_trans = len(transactions)
    report_lines.append(f"Transactions: {n_trans}")
    # Apriori
    freq = apriori(transactions, 0.02)
    rules = generate_rules(freq, 0.3)
    report_lines.append(f"Apriori — frequent itemsets: {len(freq)}, rules: {len(rules)}")
    # FP-Growth
    freq_fp = fpgrowth(transactions, 0.02)
    report_lines.append(f"FP‑Growth — frequent itemsets: {len(freq_fp)}")
    # Top 10 rules
    report_lines.append("Top‑10 rules by confidence:")
    for a,c,sup,conf,lift in rules[:10]:
        report_lines.append(f"{set(a)} -> {set(c)}  sup={sup:.3f}  conf={conf:.3f}  lift={lift:.2f}")

def clustering_analysis(points, report_lines):
    report_lines.append("\n=== Clustering Analysis (Mall Customers) ===")
    report_lines.append(f"Samples: {len(points)}")
    # KMeans and KMedians for k=2..8
    report_lines.append("k, KMeans inertia, KMedians total L1")
    for k in range(2,9):
        km = KMeans(k, random_state=0).fit(points)
        kmed = KMedians(k, random_state=0).fit(points)
        report_lines.append(f"{k}, {km.inertia_:.2f}, {kmed.inertia_:.2f}")
    # Hierarchical single‑link for k=5
    ag = AgglomerativeSingleLink(5).fit(points)
    clusters = len(set(ag.labels_))
    report_lines.append(f"Single‑link hierarchical — k=5, clusters formed: {clusters}")
    # DBSCAN heuristic eps
    db = DBSCAN(eps=15, min_samples=5).fit(points)
    n_clusters = len({lbl for lbl in db.labels_ if lbl != -1})
    noise = sum(1 for lbl in db.labels_ if lbl == -1)
    report_lines.append(f"DBSCAN eps=15, min_samples=5 — clusters: {n_clusters}, noise points: {noise}")

def main():
    start = time.time()
    download_if_missing()
    report_lines = [f"LAB2 auto‑report  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"]
    # Rule mining
    try:
        groceries = load_groceries()
        rule_mining(groceries, report_lines)
    except Exception as e:
        report_lines.append(f"[ERROR] Association rule section failed: {e}")
    # Clustering
    try:
        mall = load_mall()
        clustering_analysis(mall, report_lines)
    except Exception as e:
        report_lines.append(f"[ERROR] Clustering section failed: {e}")
    # Write report
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nReport saved to {REPORT_PATH}")
    print(f"Total time: {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
