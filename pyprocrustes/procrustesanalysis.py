import numpy as np
from numpy.random import random as rng
import matplotlib.pyplot as plt

def procrustes(k1, k2):
    flag = False
    if flag:
        plt.scatter(k1[:,0],k1[:,1],c='red', s=100)
        plt.scatter(k2[:,0],k2[:,1],c='blue', s=100)
    k1 = __translate__(k1)
    k2 = __translate__(k2)
    k1 = __scale__(k1)
    k2 = __scale__(k2)
    k1, k2 = __rotate__(k1, k2)
    if flag:
        plt.scatter(k1[:,0],k1[:,1], facecolors='none', edgecolors='red', s=300)
        plt.scatter(k2[:,0],k2[:,1], facecolors='none', edgecolors='blue', s=200)
        plt.axis('equal')
        plt.show()
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
    theta = rng()
    u = np.cos(theta) * k1[:,0] - np.sin(theta) * k1[:,1]
    v = np.sin(theta) * k1[:,0] + np.cos(theta) * k1[:,1]
    k1 = np.array([u,v]).T
    u = np.cos(theta) * k2[:,0] - np.sin(theta) * k2[:,1]
    v = np.sin(theta) * k2[:,0] + np.cos(theta) * k2[:,1]
    k2 = np.array([u,v]).T
    far1 = k1[0]

    k1__ = np.array(sorted(k1, key=lambda x: ((np.arctan2(x[1], x[0])%(2*np.pi) - np.arctan2(far1[1], far1[0])%(2*np.pi))%(2*np.pi),
                                             ((x[0] - far1[0])**2 + (x[1] - far1[1])**2) )))
    for j, k in enumerate(k2):
        k2__ = np.array(sorted(k2, key=lambda x: ((np.arctan2(x[1], x[0])%(2*np.pi) - np.arctan2(k[1], k[0])%(2*np.pi))%(2*np.pi),
                                                 ((x[0] - k[0])**2 + (x[1] - k[1])**2) )))
        num = sum(k2__[:,0] * k1__[:,1] - k2__[:,1] * k1__[:,0])
        den = sum(k2__[:,0] * k1__[:,0] + k2__[:,1] * k1__[:,1])
        theta = np.arctan2(num, den)
        u = np.cos(theta) * k2__[:,0] - np.sin(theta) * k2__[:,1]
        v = np.sin(theta) * k2__[:,0] + np.cos(theta) * k2__[:,1]
        k2__ = np.array([u,v]).T
        c = __compare__(k1__, k2__)
        if c < 1e-3:
            return k1__, k2__

    print("No match")
    return k1__, k2__

def __compare__(k1, k2):
    s = 0.0
    for i in range(len(k1)):
        s += (k2[i,0] - k1[i,0])**2
        s += (k2[i,1] - k1[i,1])**2
    return np.sqrt(s)
