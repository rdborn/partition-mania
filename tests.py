import numpy as np
import matplotlib.pyplot as plt
from pyprocrustes.procrustesanalysis import procrustes

a = np.array([[1,1],[2,1],[3,1],[2,3],[3,3],[4,1],[2,2]])
b = np.array([[8,7],[9,7],[6,7],[7,7],[8,6],[7,5],[8,5]])
print("Here we go")
k1, k2, c = procrustes(a,b)
print(c)

plt.scatter(a[:,0],a[:,1],c='red', s=100)
plt.scatter(b[:,0],b[:,1],c='blue', s=100)
plt.scatter(k1[:,0],k1[:,1], facecolors='none', edgecolors='red', s=300)
plt.scatter(k2[:,0],k2[:,1], facecolors='none', edgecolors='blue', s=200)
plt.show()
