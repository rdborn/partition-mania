import numpy as np
import pandas as pd

def procrustes(k1, k2):
    k1 = __translate__(k1)
    k2 = __translate__(k2)
    k1 = __scale__(k1)
    k2 = __scale__(k2)
    k1, k2 = __rotate__(k1, k2)
    return k1, k2, __compare__(k1, k2)

def __translate__(k):
    mu = np.mean(k, axis=0)
    return 1.0 * k - mu

def __scale__(k):
    sigma = np.std(k, axis=0)
    return 1.0 * k / sigma

def __rotate__(k1, k2):
    num = 0.0
    den = 0.0
    k1 = np.array(sorted(k1, key=lambda x: (x[0]**2 + x[1]**2)))
    k2 = np.array(sorted(k2, key=lambda x: (x[0]**2 + x[1]**2)))
    for i in range(len(k1)):
        num += k2[i,0] * k1[i,1] - k2[i,1] * k1[i,0]
        den += k2[i,0] * k1[i,0] + k2[i,1] * k1[i,1]
    theta = np.arctan2(num, den)
    u = np.cos(theta) * k2[:,0] - np.sin(theta) * k2[:,1]
    v = np.sin(theta) * k2[:,0] + np.cos(theta) * k2[:,1]
    k2 = np.array([u,v]).T
    return k1, k2

def __compare__(k1, k2):
    # k1 = np.array(sorted(k1, key=lambda x: (x[0], x[1])))
    # k2 = np.array(sorted(k2, key=lambda x: (x[0], x[1])))
    return np.sqrt(np.sum((k2-k1)**2))
