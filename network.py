import igraph as ig
import random

N = 1_00_000
p = 0.5
g1 = ig.Graph.Erdos_Renyi(n=N, p=p, directed=False, loops=False)
for e in g1.es:
    print(e.source, e.target)
