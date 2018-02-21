import numpy as np
from numpy.random import random as rng
import matplotlib.pyplot as plt

class SetOfSets:
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
        idx = np.inf
        while idx >= len(self.edges):
            idx = np.int(np.floor(rng()*(len(self.edges))))
            if idx >= len(self.edges):
                if idx == 0:
                    return [-1,-1]
        return self.edges[idx]

    def next_edge(self, n):
        if len(self.edges) == 0:
            return [-1, -1]
        return self.edges[n%len(self.edges)]

    def random_adjacent_edge(self, x):
        adj_edges = self.adjacent_edges(x)
        idx = np.int(np.floor(rng()*(len(adj_edges))))
        print(adj_edges)
        print(idx)
        return adj_edges[idx]

    def adjacent_edges(self, x):
        u = self.find(x)
        adj_edges = []
        for n in self.nodes:
            if self.find(n) == u:
                for i, adjacent in enumerate(self.adjacency[n]):
                    if adjacent:
                        if i < n:
                            adj_edges.extend([[i,n]])
                        else:
                            adj_edges.extend([[n,i]])
        return adj_edges

    def remove_edge(self, e):
        self.edges.remove(e)
        self.adjacency[e[0],e[1]] = 0
        self.adjacency[e[1],e[0]] = 0
        return True

    def plot(self):
        x = np.array(self.nodes) % self.dx
        y = self.dx - np.array(self.nodes) / self.dx
        colors = np.zeros(len(self.nodes))
        idxs = np.array(range(len(self.sets)))
        sets = -np.ones(len(self.sets))
        j = 0
        coords = []
        for i, n in enumerate(self.nodes):
            u = self.find(n)
            if not u in sets:
                sets[j] = u
                j += 1
                coords.extend([[]])
            colors[i] = idxs[sets == u]
            coords[idxs[sets == u]].extend([[n % self.dx, self.dx - n / self.dx]])
        for i, c in enumerate(coords):
            points = np.array(c).T
            plt.plot(points[0],points[1])
        plt.scatter(x, y, c=colors%9, cmap='Set1', s=1000)
        plt.show()

    def get_setmates(self, x):
        a = self.find(x)
        nodes = []
        for n in self.nodes:
            if self.find(n) == a:
                nodes.extend([n])
        return np.array(nodes)

    def lock_set(self, a):
        nodes = []
        for n in self.nodes:
            if self.find(n) == a:
                nodes.extend([n])
                north = [n - self.dx, n]
                east = [n, n + 1]
                south = [n, n + self.dx]
                west = [n - 1, n]
                edges = [north, east, south, west]
                for e in edges:
                    if e in self.edges:
                        self.remove_edge(e)
        return np.array(nodes)

    def elligible_pockets(self):
        n_neighbors = np.sum(self.adjacency, axis=0)
        return np.array(self.nodes)[n_neighbors < 3]

    def coords_to_node(self, a):
        n1 = self.dx * (self.dx - (a[1] + 1)) + a[0] + 1
        return n1

    def remove_edge_if_it_exists(self, a, b):
        if self.edge_exists(a, b):
            n1 = self.coords_to_node(a)
            n2 = self.coords_to_node(b)
            if n1 < n2:
                self.remove_edge([n1, n2])
            else:
                self.remove_edge([n2, n1])
            return True
        return False

    def edge_exists(self, a, b):
        if len(a) == 2:
            n1 = self.coords_to_node(a)
        else:
            n1 = a
        if len(b) == 2:
            n2 = self.coords_to_node(b)
        else:
            n2 = b
        if n1 == n2:
            return False
        if n1 < n2:
            return [n1, n2] in self.edges
        else:
            return [n2, n1] in self.edges
