import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysets.pysets import SetOfSets

N = 64
dx = 8
maxW = 7
while(True):
    Weights = [2,3,5,5,5,5,5,6,6,7,7,7]
    S = SetOfSets(N, dx)
    while len(S.edges) > 0:
        e = S.random_edge()
        u = S.find(e[0])
        v = S.find(e[1])
        if u == v:
            S.remove_edge(e)
        # elif S.weight[u] in Weights:
        #     S.remove_edge(e)
        # elif S.weight[v] in Weights:
        #     S.remove_edge(e)
        else:
            S.union(u, v)
            uUv = S.find(u)
            w = S.weight[uUv]
            if w in Weights:
                Weights.remove(w)
                S.lock_set(uUv)
            elif w > maxW:
                break
    print(S.sets)
    print(Weights)
    if len(Weights) == 0:
        break
S.plot()
