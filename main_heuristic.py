import numpy as np
import matplotlib.pyplot as plt
from pysets.pysets import SetOfSets
from pyheuristic.fitness import place_next_piece
from constants import Shapes, BigShapes, N, dx, Weights

while(True):
    S = SetOfSets(N, dx)
    S__ = place_next_piece(S, BigShapes)
    # S__ = place_next_piece(S, Shapes)
    S__.plot()
