
"""Demo script that runs Apriori + FP‑Growth on Groceries dataset
and K‑Means clustering on Mall Customers dataset.

Usage:
    python -m src.demo
"""
import csv, random, os, math
from .apriori import apriori, generate_rules
from .fpgrowth import fpgrowth
from .clustering import KMeans

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")

def load_groceries():
    path = os.path.join(DATA_DIR, "groceries.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("groceries.csv not found in data/. Download from Kaggle.")
    trans = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            trans.append(set(row))  # кожен рядок – список продуктів у транзакції
    return trans

def load_mall():
    path = os.path.join(DATA_DIR, "Mall_Customers.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Mall_Customers.csv not found in data/. Download from Kaggle.")
    pts = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row["Annual Income (k$)"]), float(row["Spending Score (1-100)"])))
    return pts

def main():
    print("=== Association rules ===")
    trans = load_groceries()
    freq = apriori(trans, 0.02)         # 2 % support
    rules = generate_rules(freq, 0.3)   # 30 % confidence
    print("Total rules:", len(rules))
    print("Top‑5 rules:")
    for a,c,sup,conf,lift in rules[:5]:
        print(f"{set(a)} → {set(c)}  sup={sup:.3f}  conf={conf:.3f}  lift={lift:.2f}")

    print("\n=== FP‑Growth ===")
    freq_fp = fpgrowth(trans, 0.02)
    print("Frequent itemsets (FP):", len(freq_fp))

    print("\n=== K‑Means clustering ===")
    data = load_mall()
    for k in range(2,6):
        km = KMeans(k, random_state=0).fit(data)
        print(f"k={k}, inertia={km.inertia_:.2f}")

if __name__ == "__main__":
    main()
