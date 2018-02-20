import numpy as np
import pandas as pd
from numpy.random import random as rng
import matplotlib.pyplot as plt

class SetOfSets:
    # self.up
    # self.weight
    # self.sets
    # self.edges
    # self.nodes
    # self.adjacency
    def __init__(self, N, dx):
        self.nodes = range(N)
        self.weight = np.ones(N)
        self.up = range(N)
        self.sets = set(self.nodes)
        self.edges = []
        self.adjacency = np.zeros([N, N])
        self.__generate_edges__(dx)
        self.dx = dx

    def __generate_edges__(self, dx):
        N = len(self.nodes)
        for n in self.nodes:
            right = n + 1
            down = n + dx
            right_exists = (((n % dx) + 1) % dx > (n % dx))
            down_exists = (n + dx < N)
            if right_exists:
                self.edges.extend([[n, right]])
                self.adjacency[n, right] = 1
                self.adjacency[right, n] = 1
            if down_exists:
                self.edges.extend([[n, down]])
                self.adjacency[n, down] = 1
                self.adjacency[down, n] = 1
        return True

    def find(self, x):
        parent = self.up[x]
        while parent != self.up[parent]:
            parent = self.up[parent]
        self.compress(x, parent)
        return parent

    def compress(self, child, elder):
        parent = self.up[child]
        while parent != elder:
            self.up[child] = elder
            child = parent
            parent = self.up[parent]
        return True

    def union(self, a, b):
        if not (a in self.sets and b in self.sets):
            print("Union can only be generated from roots. No action taken.")
            return False
        if self.weight[a] > self.weight[b]:
            child = b
            adopter = a
        else:
            child = a
            adopter = b
        self.up[child] = adopter
        self.weight[adopter] += self.weight[child]
        self.sets -= set([child])
        return True

    def random_edge(self):
        return self.edges[np.int(rng()*len(self.edges))]

    def remove_edge(self, e):
        self.edges.remove(e)
        return True

    def plot(self):
        x = np.array(self.nodes) % self.dx
        y = self.dx - np.array(self.nodes) / self.dx
        colors = np.zeros(len(self.nodes))
        idxs = np.array(range(len(self.sets)))
        sets = -np.ones(len(self.sets))
        j = 0
        for i, n in enumerate(self.nodes):
            u = self.find(n)
            if not u in sets:
                sets[j] = u
                j += 1
            colors[i] = idxs[sets == u]
        plt.scatter(x, y, c=colors%9, cmap='Set1_r', s=1000)
        plt.show()

    def lock_set(self, a):
        for n in self.nodes:
            if self.find(n) == a:
                north = [n - self.dx, n]
                east = [n, n + 1]
                south = [n, n + self.dx]
                west = [n - 1, n]
                edges = [north, east, south, west]
                for e in edges:
                    if e in self.edges:
                        self.remove_edge(e)
