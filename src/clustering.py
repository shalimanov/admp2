
"""Simple clustering algorithms (LAB2/src/clustering.py)"""
import math, random, itertools, statistics, collections

def _eucl(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def _manh(a,b): return sum(abs(x-y) for x,y in zip(a,b))

class KMeans:
    def __init__(self,k,max_iter=100,tol=1e-4,random_state=None):
        self.k=k; self.max_iter=max_iter; self.tol=tol
        if random_state is not None: random.seed(random_state)
    def fit(self,data):
        self.centroids=random.sample(list(data),self.k)
        for _ in range(self.max_iter):
            clusters=[[] for _ in range(self.k)]
            for p in data:
                idx=min(range(self.k), key=lambda i:_eucl(p,self.centroids[i]))
                clusters[idx].append(p)
            new=[]
            for cl in clusters:
                if cl: new.append(tuple(sum(col)/len(cl) for col in zip(*cl)))
                else: new.append(random.choice(data))
            shift=max(_eucl(a,b) for a,b in zip(self.centroids,new))
            self.centroids=new
            if shift<self.tol: break
        self.inertia_=sum(_eucl(p,self.centroids[min(range(self.k), key=lambda i:_eucl(p,self.centroids[i]))])**2 for p in data)
        return self
    def predict(self,points):
        return [min(range(self.k), key=lambda i:_eucl(p,self.centroids[i])) for p in points]

class KMedians(KMeans):
    def fit(self,data):
        self.centroids=random.sample(list(data),self.k)
        for _ in range(self.max_iter):
            clusters=[[] for _ in range(self.k)]
            for p in data:
                idx=min(range(self.k), key=lambda i:_manh(p,self.centroids[i]))
                clusters[idx].append(p)
            new=[]
            for cl in clusters:
                if cl: new.append(tuple(statistics.median(col) for col in zip(*cl)))
                else: new.append(random.choice(data))
            shift=max(_manh(a,b) for a,b in zip(self.centroids,new))
            self.centroids=new
            if shift<self.tol: break
        self.inertia_=sum(_manh(p,self.centroids[min(range(self.k), key=lambda i:_manh(p,self.centroids[i]))]) for p in data)
        return self

class AgglomerativeSingleLink:
    def __init__(self,k): self.k=k
    def fit(self,data):
        clusters=[{i} for i in range(len(data))]
        d={(i,j):_eucl(data[i],data[j]) for i in range(len(data)) for j in range(i+1,len(data))}
        while len(clusters)>self.k:
            i_min=j_min=None; best=float('inf')
            for a,b in itertools.combinations(range(len(clusters)),2):
                dist=min(d[tuple(sorted((i,j)))] for i in clusters[a] for j in clusters[b])
                if dist<best: best=dist;i_min,j_min=a,b
            clusters[i_min]|=clusters[j_min]
            del clusters[j_min]
        self.labels_=[None]*len(data)
        for lbl,cl in enumerate(clusters):
            for idx in cl: self.labels_[idx]=lbl
        return self

class DBSCAN:
    def __init__(self,eps,min_samples): self.eps=eps; self.min_samples=min_samples
    def _region(self,data,i): return [j for j,p in enumerate(data) if _eucl(p,data[i])<=self.eps]
    def fit(self,data):
        n=len(data); UNVIS=0; NOISE=-1
        visited=[UNVIS]*n; self.labels_=[None]*n; cid=0
        for i in range(n):
            if visited[i]: continue
            visited[i]=1
            neigh=self._region(data,i)
            if len(neigh)<self.min_samples:
                self.labels_[i]=NOISE
            else:
                self._expand(data,i,neigh,cid,visited)
                cid+=1
        return self
    def _expand(self,data,i,neigh,cid,visited):
        self.labels_[i]=cid
        q=collections.deque(neigh)
        while q:
            j=q.popleft()
            if not visited[j]:
                visited[j]=1
                j_neigh=self._region(data,j)
                if len(j_neigh)>=self.min_samples:
                    q.extend(j_neigh)
            if self.labels_[j] is None:
                self.labels_[j]=cid
