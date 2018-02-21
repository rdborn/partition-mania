import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random as rng
from pysets.pysets import SetOfSets
from pyprocrustes.procrustesanalysis import procrustes
from copy import deepcopy
from constants import Shapes, N, dx, Weights


tries = 0
while(True):
    tries += 1
    if tries % 100 == 0:
        print("Still going")
        print(tries)
    w = deepcopy(Weights)
    S = SetOfSets(N, dx)
    while len(S.sets) > len(Shapes):
        e = S.random_edge()
        if (np.array(e) < 0).any():
            break
        else:
            u = S.find(e[0])
            v = S.find(e[1])
            if u == v:
                S.remove_edge(e)
                n_nodes = len(S.get_setmates(u))
                if n_nodes in w:
                    w.remove(n_nodes)
                    S.lock_set(S.find(u))
            else:
                if len(S.get_setmates(u)) + len(S.get_setmates(v)) > np.max(Weights):
                    S.remove_edge(e)
                else:
                    S.union(u, v)
                    S.remove_edge(e)
    if (np.array(e) < 0).any():
        pass
    else:
        partition_weights = np.zeros(len(Weights))
        nodes = []
        for i, set in enumerate(S.sets):
            nodes_in_set_i = S.get_setmates(set)
            nodes.append([nodes_in_set_i])
            partition_weights[i] = len(S.get_setmates(set))
        partition_weights.sort()
        flag = True
        for i, pweight in enumerate(partition_weights):
            flag &= (abs(pweight - 1.0*Weights[i]) < 1e-3)
            if not flag:
                break
        if flag:
            Shapes__ = deepcopy(Shapes)
            for set in S.sets:
                nodes = S.get_setmates(set)
                x = nodes % dx
                y = dx - nodes / dx
                set_i = np.array([x,y]).T
                minval = np.inf
                best_shape = Shapes__[0]
                if len(set_i) > 2:
                    for shape in Shapes__:
                        if len(shape) == len(set_i):
                            trash, trash, c = procrustes(np.array(shape), set_i)
                            if c < minval:
                                minval = c
                                best_shape = shape
                                if minval < 1e-3:
                                    break
                    flag &= (minval < 1e-3)
                    if minval < 1e-3:
                        Shapes__.remove(best_shape)
                    print("Weight: " + str(len(nodes)) + ", " + str(minval))
            if not flag:
                print("Nope")
            else:
                S.plot()
