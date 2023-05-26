# MIT License 
# Copyright (c) - 2023
# Fatih Altindis⁺ꜝ and Marco Congedo ꜝ
# ⁺ Abdullah Gul University, Kayseri
# ꜝ GIPSA-lab, CNRS, University Grenoble Alpes




import numpy as np
from numpy.random import Generator, PCG64
import scipy
from copy import deepcopy


# boot_all is in dimension of (feature_size x n_class*n_of_boots).
# For example if each feature vector's length is 200 and there are 2 classes,
# returned bootstrap will have 200 x 300 dimension.
def create_bootstrap(features, labels, bootsize = 25, n_of_boots = 100, w = "u", random_state = 1):
    # Create pseudo-random seeds for random integer generator
    seeds = Generator(PCG64(random_state)).integers(1,1e12, n_of_boots)
    classes = np.unique(labels)
    boot_all = []
    for c in classes:
        if w == "u":
            weights = np.ones(bootsize) / bootsize
        else:
            weights = Generator(PCG64(random_state)).random(bootsize)
            weights = weights / np.sum(weights)
        
        temp_boot = []
        temp = features[:,labels==c]
        [temp_boot.append(temp[:,Generator(PCG64(seeds[x])).integers(1,temp.shape[1],bootsize)].dot(weights)) for x in range(n_of_boots)]
        boot_all.append(np.array(temp_boot))
        
    return np.transpose(np.concatenate(boot_all))



def whiten_data(bootstraps, wh_type="smart", white_dim=16, smart_subspace=16, verbose=True):
    # Initialize whitening matrices
    M = len(bootstraps)
    D = white_dim
    L = bootstraps[0].shape[1]
    
    boldT = [np.zeros((D,L)) for m in range(M)]
    boldS = [None for m in range(M)]
    boldW = [np.zeros((D,D)) for m in range(M)]

    # Whitening begins
    verbose and print("Pre-whitening started...")
    if wh_type == "svd":
        for m in range(M):
            # order of svd return >>> U, S, VH
            boldS[m] = scipy.linalg.svd(bootstraps[m])
            boldW[m] = boldS[m][0][:,0:white_dim] * np.transpose(1/boldS[m][1][0:white_dim])
            boldT[m] = boldW[m].T @ bootstraps[m]
            verbose and print("Completed..... %" + str(100*m/M))
        verbose and print("Pre-whitening is completed with >>>  " + wh_type)
    elif wh_type == "smart":
        temp_T = [None for m in range(M)]
        boldC = [None for m in range(M)]
        for m in range(M):
            boldS[m] = scipy.linalg.svd(bootstraps[m])
            temp_T[m] = (boldS[m][0][:,0:white_dim] @ np.diag(boldS[m][1][0:white_dim])).T @ bootstraps[m]
            boldC[m] = [np.zeros((D,D)) for m in range(M)]
        verbose and print("Pre-whitening step-1 is completed...")

        for m in range(M):
            for i in range(m+1,M):
                boldC[i][m] = temp_T[i] @ temp_T[m].T
        verbose and print("Pre-whitening step-2 is completed...")

        H = np.zeros((D,D))
        for m in range(M):
            H = np.zeros((D,D))
            for j in range(M):
                if m>j:
                    H += boldC[m][j]
                if m<j:
                    H += boldC[j][m].T
            F = scipy.linalg.svd(H)
            boldW[m] = F[0][:,0:smart_subspace]
        verbose and print("Pre-whitening matrices are estimated...")

        for m in range(M):
            boldT[m] = boldW[m].T @ temp_T[m]
        verbose and print("Pre-whitening is completed with >>>  " + wh_type)
    else:
        boldT = bootstraps
        verbose and print("No pre-whitening is applied to bootstraps!")
    return boldT, boldS, boldW


def norm_U(boldU, norm_type = "unit", boldT = []):
    
    M = len(boldU)
    if norm_type == "unit":
        new_boldU = [boldU[m] / scipy.linalg.norm(boldU[m], axis=0) for m in range(M)]
        
    elif norm_type == "white":
        if not boldT:
            raise ValueError("boldT cannot be empty for smart whitening!!!")
        if len(boldU) != len(boldT):
            raise ValueError("boldU and boldT list should have same length!!!")
        new_boldU = []
        for m in range(M):
            tempU = boldU[m]
            P = boldT[m] @ boldT[m].T
            bk = np.array([np.sqrt(tempU[:,k].T @ P @ tempU[:,k]) for k in range(tempU.shape[0])])
            new_boldU.append(tempU / bk.T)
    else:
        print("boldU matrices are not normalized and returned as they are.")
        print("white or unit norm type can be chosen for normalization.")

    return new_boldU

def estimate_B(boldU, boldW, wh_type = "smart", white_dim = 16, reverse_selection = False, boldS = []):
    if len(boldU) != len(boldW):
        raise ValueError("Number of alignment matrices is not equal to number of whitening matrices!!!")
    if (wh_type == "smart") & (not boldS):
        raise ValueError("boldS can not be empty for smart whitening!!!")
    
    M = len(boldU)
    B = []

    if wh_type == "svd":
        B = [boldW[m] @ boldU[m] for m in range(M)]
    elif wh_type == "smart":
        B = [boldS[m][0][:,0:white_dim] @ np.diag(boldS[m][1][0:white_dim]) @ boldW[m] @ boldU[m] for m in range(M)]
    else:
        print("The chosen whitening type is not proper. No B matrices estimated.")
        print("Make sure the whitening type is either \"smart\" or \"svd\".")
    return B

def align_features(train_split, test_split, B, sub_dim = None):
    if sub_dim is None:
        sub_dim = B[0].shape[1]
    
    aligned_train = [B[m][:,0:sub_dim].T @ f for m, f in enumerate(train_split)]
    aligned_test = [B[m][:,0:sub_dim].T @ f for m, f in enumerate(test_split)]

    return aligned_train, aligned_test


# shape of the feature matrix should be in the order of
# features X trials

def zero_mean_vecs(vecs):
    return np.transpose(deepcopy(vecs.T) - np.mean(vecs, axis=1))

def norm_vecs(vecs):
    return deepcopy(vecs) / np.mean(scipy.linalg.norm(vecs, axis=0))

def cross_cov(X):
    M = len(X)
    C = [[[] for i in range(M)] for j in range(M)]
    for i in range(M):
        for j in range(i,M):
            C[i][j] = X[i] @ X[j].T / X[i].shape[1]
            C[j][i] = C[i][j].T
    return C

def flipcol(boldU, M, n, e):
    U = deepcopy(boldU)
    for i in range(M):
        temp = deepcopy(U[i][:,e])
        U[i][:,e] = deepcopy(U[i][:,n])
        U[i][:,n] = deepcopy(temp)
    return U


def flip_and_permute(boldU, boldX, M, k, input_="d", covest="SCM", dims=1, meanX=0, trace=False):
    # if the input_ is "d", SCM cross-covariances are estimated
    if input_ == "d":
        boldC = cross_cov(boldX)
    else:
        boldC = boldX
    
    U = deepcopy(boldU)
    udim = U[0].shape[0]
    D = [[] for i in range(M)]
    for i in range(M):
        for j in range(M):
            D[i].append(np.diag([U[i][:,n].T @ boldC[i][j] @ U[j][:,n] for n in range(udim)]))
            
    for e in range(udim):
        maxx = 0
        # find the position of the absolute maximum
        for i in range(M-1):
            for j in range(i+1,M):
                for n in range(e,udim):
                    absd = abs(D[i][j][n,n])
                    if absd > maxx:
                        maxx = deepcopy(absd)
                        p = deepcopy([i, j, n])
        
        # flip sign of boldU[j][n,n] if absolute maximum is negative
        i, j, n = p[0], p[1], p[2]
        if D[i][j][n,n] < 0:
            U[j][:,n] *= -1
        
        # flip sign of boldU[j] for all j≠i, i=1:M if their corresponding element is negative
        for x in range(M):
            if x != j:
                if D[i][x][n,n] < 0:
                    U[x][:,n] *= -1
        
        # bring the maximum from position n to on top (for current e)
        if n != e:
            U = flipcol(U, M, n, e)
        
        # compute D again
        D = [[] for i in range(M)]
        for j in range(M):
            for i in range(M):
                D[i].append(np.diag([U[i][:,n].T @ boldC[i][j] @ U[j][:,n] for n in range(udim)]))

    return U

