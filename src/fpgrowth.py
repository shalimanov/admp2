
"""Very small FP‑Growth implementation (LAB2/src/fpgrowth.py)"""
from collections import Counter

class _Node:
    __slots__ = ("item","cnt","parent","children","next")
    def __init__(self, item, parent):
        self.item=item; self.cnt=1; self.parent=parent
        self.children={}; self.next=None

def _add_path(items, node, header):
    if not items: return
    first,*rest = items
    child = node.children.get(first)
    if child:
        child.cnt += 1
    else:
        child = _Node(first,node)
        node.children[first] = child
        if header[first][1] is None:
            header[first][1] = child
        else:
            cur = header[first][1]
            while cur.next: cur=cur.next
            cur.next = child
    _add_path(rest, child, header)

def _build_tree(transactions, min_cnt):
    C = Counter()
    for t in transactions:
        C.update(t)
    freq_items = {i for i,c in C.items() if c>=min_cnt}
    if not freq_items: return None,None
    header = {i:[C[i],None] for i in freq_items}
    root = _Node(None,None)
    for t in transactions:
        filtered = [i for i in t if i in freq_items]
        filtered.sort(key=lambda x: header[x][0], reverse=True)
        _add_path(filtered, root, header)
    return root, header

def _asc(node):
    path=[]
    while node and node.parent and node.parent.item is not None:
        node=node.parent
        path.append(node.item)
    return path

def _mine(header,min_cnt,prefix,freq,n):
    # items by ascending freq
    for item, (cnt,node) in sorted(header.items(), key=lambda x: x[1][0]):
        new_prefix = prefix | {item}
        freq[frozenset(new_prefix)] = cnt/n
        cond_pat=[]
        while node:
            cond_pat += [_asc(node)]*node.cnt
            node = node.next
        subtree, sub_header = _build_tree(cond_pat, min_cnt)
        if sub_header:
            _mine(sub_header,min_cnt,new_prefix,freq,n)

def fpgrowth(transactions, min_support):
    n=len(transactions); min_cnt=int(min_support*n+0.999)
    root,header=_build_tree(transactions,min_cnt)
    freq={}
    if header: _mine(header,min_cnt,set(),freq,n)
    return freq
