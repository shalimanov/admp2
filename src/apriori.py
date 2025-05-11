
"""Apriori algorithm & rule generation (LAB2/src/apriori.py)"""
from itertools import combinations
from collections import defaultdict
import math

def _support_cnt(transactions, itemset):
    return sum(1 for t in transactions if itemset.issubset(t))

def apriori(transactions, min_support):
    n = len(transactions)
    min_cnt = math.ceil(min_support * n)
    # L1
    item_cnt = defaultdict(int)
    for t in transactions:
        for itm in t:
            item_cnt[frozenset([itm])] += 1
    Lk = {it: c/n for it, c in item_cnt.items() if c >= min_cnt}
    all_freq = dict(Lk)
    k = 2
    while Lk:
        prev = list(Lk)
        cand = set()
        for i in range(len(prev)):
            for j in range(i+1, len(prev)):
                union = prev[i] | prev[j]
                if len(union) == k and all((union - frozenset([x])) in Lk for x in union):
                    cand.add(union)
        Lk = {}
        for c in cand:
            cnt = _support_cnt(transactions, c)
            if cnt >= min_cnt:
                Lk[c] = cnt / n
        all_freq.update(Lk)
        k += 1
    return all_freq

def generate_rules(freq_itemsets, min_confidence):
    rules = []
    for itemset, supp in freq_itemsets.items():
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                conf = supp / freq_itemsets[antecedent]
                if conf >= min_confidence:
                    lift = conf / freq_itemsets[consequent]
                    rules.append((antecedent, consequent, supp, conf, lift))
    rules.sort(key=lambda r: (r[3], r[4]), reverse=True)
    return rules
