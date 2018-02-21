import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random as rng
from pysets.pysets import SetOfSets
from pyprocrustes.procrustesanalysis import procrustes
from copy import deepcopy

def convex_hull(shape):
    shape = np.array(shape)
    minx = np.min(shape[:,0])
    maxx = np.max(shape[:,0])
    miny = np.min(shape[:,1])
    maxy = np.max(shape[:,1])
    cvx_hull = np.zeros([len(range(minx, maxx+1)) * len(range(miny, maxy+1)), 2])
    i = 0
    for x in range(minx, maxx+1):
        for y in range(miny, maxy+1):
            cvx_hull[i] = np.array([x, y])
            i += 1
    return cvx_hull.T

def check_internal_edges(S, shape, cvx_hull, anchor, pocket):
    shape = np.array(shape)
    cvx_hull = cvx_hull - anchor
    cvx_hull = cvx_hull + pocket
    shape = shape - anchor
    shape = shape + pocket
    flag = True
    for i, n1 in enumerate(shape):
        for j, n2 in enumerate(shape):
            d = np.abs(np.sqrt(np.sum((n1 - n2)**2)))
            if abs(d - 1) < 1e-3:
                flag &= S.edge_exists(np.round(n1), np.round(n2))
    return flag

def wall_directions(S, pocket):
    north = pocket + np.array([0,1])
    east = pocket + np.array([1,0])
    south = pocket + np.array([0,-1])
    west = pocket + np.array([-1,0])
    idxs = np.zeros(4, dtype=bool)
    idxs[0] = S.edge_exists(pocket, north)
    idxs[1] = S.edge_exists(pocket, east)
    idxs[2] = S.edge_exists(pocket, south)
    idxs[3] = S.edge_exists(pocket, west)
    return idxs

def dist_2_walls(S, pocket, shape, anchor):
    shape = np.array(shape)
    idxs = wall_directions(S, pocket)
    north = idxs[0]
    east = idxs[1]
    south = idxs[2]
    west = idxs[3]
    sum_sq_dist = 0
    n_d = 0
    if north:
        for p in shape:
            if not S.edge_exists(np.array([p[0], pocket[1]]), np.array([p[0], pocket[1] + 1])):
                sum_sq_dist += (pocket[1] - p[1])**2
                n_d += 1
    if east:
        for p in shape:
            if not S.edge_exists(np.array([pocket[0], p[1]]), np.array([pocket[0]+1, p[1]])):
                sum_sq_dist += (pocket[0] - p[0])**2
                n_d += 1
    if south:
        for p in shape:
            if not S.edge_exists(np.array([p[0], pocket[1]]), np.array([p[0], pocket[1] - 1])):
                sum_sq_dist += (p[1] - pocket[1])**2
                n_d += 1
    if west:
        for p in shape:
            if not S.edge_exists(np.array([pocket[0], p[1]]), np.array([pocket[0]-1, p[1]])):
                sum_sq_dist += (p[0] - pocket[0])**2
                n_d += 1
    return np.sqrt(sum_sq_dist / n_d) if n_d != 0 else np.inf

def investigate_shape(S, shape, pocket):
    shape = np.array(shape)
    cvx_hull = convex_hull(shape).T
    angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    rmsd = np.inf * np.ones([len(cvx_hull),len(angles)])
    for i, theta in enumerate(angles):
        u_cvx = np.cos(theta) * cvx_hull[:,0] - np.sin(theta) * cvx_hull[:,1]
        v_cvx = np.sin(theta) * cvx_hull[:,0] + np.cos(theta) * cvx_hull[:,1]
        u_shape = np.cos(theta) * shape[:,0] - np.sin(theta) * shape[:,1]
        v_shape = np.sin(theta) * shape[:,0] + np.cos(theta) * shape[:,1]
        cvx_hull__ = np.array([u_cvx, v_cvx]).T
        shape__ = np.array([u_shape, v_shape]).T
        for j, anchor in enumerate(cvx_hull__):
            if check_internal_edges(S, shape__, cvx_hull__, anchor, pocket):
                rmsd[j, i] = dist_2_walls(S, pocket, shape, anchor)
    return rmsd

def investigate_pocket(S, Shapes, pocket):
    best_rmsd = np.inf
    best_shape = Shapes[0]
    for shape in Shapes:
        rmsd = investigate_shape(S, shape, pocket)
        if np.min(rmsd) < np.min(best_rmsd):
            best_rmsd = rmsd
            best_shape = shape
    return best_shape, best_rmsd

def investigate_board(S, Shapes):
    elligible_pockets = S.elligible_pockets()
    best_rmsd = np.inf
    best_shape = Shapes[0]
    best_pocket = np.zeros(2)
    for node in elligible_pockets:
        x = node % S.dx
        y = S.dx - node / S.dx - 1
        pocket = np.array([x,y])
        shape, rmsd = investigate_pocket(S, Shapes, pocket)
        if np.min(rmsd) < np.min(best_rmsd):
            best_rmsd = rmsd
            best_shape = shape
            best_pocket = pocket
    best_angle = 0.0
    best_anchor = best_shape[0]
    for i in range(4):
        for j, anchor in enumerate(convex_hull(best_shape).T):
            if (best_rmsd == np.min(best_rmsd))[j,i]:
                best_angle = np.pi * i / 2
                best_anchor = anchor
                break
    return best_pocket, best_shape, best_angle, best_anchor, best_rmsd

def place_piece(S, shape, pocket, angle, anchor):
    shape = np.array(shape)
    u = np.cos(angle) * shape[:,0] - np.sin(angle) * shape[:,1]
    v = np.sin(angle) * shape[:,0] + np.cos(angle) * shape[:,1]
    piece = np.array([u,v]).T - anchor + pocket
    for n1 in piece:
        for n2 in piece:
            if S.remove_edge_if_it_exists(n1, n2):
                a = np.int(S.coords_to_node(n1))
                b = np.int(S.coords_to_node(n2))
                if S.find(a) != S.find(b):
                    S.union(S.find(a), S.find(b))
    node = S.coords_to_node([np.int(piece[0][0]), np.int(piece[0][1])])
    S.lock_set(S.find(node))
    return S

def place_next_piece(S, Shapes):
    S__ = deepcopy(S)
    Shapes__ = deepcopy(Shapes)
    print(len(Shapes__))
    if len(Shapes__) == 0:
        return S__
    best_pocket, best_shape, best_angle, best_anchor, best_rmsd = investigate_board(S__, Shapes__)
    S__ = place_piece(S__, best_shape, best_pocket, best_angle, best_anchor)
    Shapes = Shapes__.remove(best_shape)
    S__.plot()
    # print(S__.up)
    S__ = place_next_piece(S__, Shapes__)
    return S__
