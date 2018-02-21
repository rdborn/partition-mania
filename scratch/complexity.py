import numpy as np
from copy import deepcopy

N = 64
Weights = [7,7,7,6,6,5,5,5,5,5,3,2,1]
r = 1
t = 1
for w in Weights:
    r *= 4
    t *= (np.sqrt(N) - np.sqrt(w) + 1)**2
    N -= w

print(t*r)

r = np.array([1,2,4,4,4,4,1,4,2,4,4,4,4])

print(t*np.prod(r))

################

t = np.array([64,56,49,42,36,35,36,36,42,30,36,36,36])
print(np.prod(r*t))

################

# Number of edges each original shape shares with its convex hull
possibles = [   [1],
                [1,2],
                [1,2],
                [1,2,3],
                [1,3],
                [1,4],
                [1],
                [1,3],
                [2,3],
                [1],
                [1,2,3],
                [1,2,3],
                [1,2,3]]

def subset_sum(possibles, target, partial):
    s = sum(partial)
    if s == target:
        return 1
    if s > target:
        return 0
    return_val = 0
    for i in range(len(possibles)):
        p = possibles[i]
        remaining = possibles[i+1:]
        for n in p:
            return_val += subset_sum(remaining, target, partial + [n])
    return return_val

print(subset_sum(possibles, 8, []))
