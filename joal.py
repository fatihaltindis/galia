# MIT License 
# Copyright (c) - 2023
# Fatih Altindis⁺ꜝ and Marco Congedo ꜝ
# ⁺ Abdullah Gul University, Kayseri
# ꜝ GIPSA-lab, CNRS, University Grenoble Alpes



import numpy as np
import scipy
from copy import deepcopy



def trifact(P, check = True):
    from copy import deepcopy
    if len(np.shape(P)) == 2:
        if not all(np.shape(P)):
            raise ValueError("Input matrix (P) must be square!") 
        else:
            n = np.shape(P)[0]
    else:
        raise ValueError("Input matrix (P) must be 2 dimensional and square!") 

    for j in range(n-1):
        f = deepcopy(P[j,j])
        P[j,j] = np.sqrt(f)
        g = deepcopy(P[j,j])
        for i in range(j+1,n):
            theta = P[i,j] / f
            c_theta = np.conj(theta)
            for k in range(i,n):
                P[k,i] -= c_theta * P[k,j]
            P[i,j] = theta * g
            P[j,i] = 0.
    P[n-1,n-1] = np.sqrt(P[n-1,n-1])
    
    return P


def normalize_col(U):
    normalized_U = U / scipy.linalg.norm(U, axis=0)
    return normalized_U

def joint_alignment(boldT, U_init=None, tol=1e-8, maxiter=1000, verbose=False):
    from copy import deepcopy
    from scipy import linalg
    import warnings

    iter_, conv, oldconv, conv_ = 1, 0.0, 1.01, 0.0
    M, D = len(boldT), boldT[0].shape[0]

    # Computation of input data C_ijk (only lower triangular part of matrix of matrices)
    # This is the pair-wise cross-covarainces of the given M domains
    C = [[np.zeros((D,D)) for i in range(j+1)] for j in range(M)]
    for j in range(M):
        for i in range(j+1,M):
            C[i][j] = boldT[i] @ boldT[j].T

    # Initialize U with identity
    if U_init is None:
        boldU = [np.eye(D) for i in range(M)]
    else:
        if all([u.shape == (D,D) for u in U_init]) and len(U_init) == M:
            boldU = U_init
        else:
            warnings.warn("Dimensionality of the provided U matrices are not compatible for initialization!")
            warnings.warn("U matrices are initialized with Identity matrices.")
            boldU = [np.eye(D) for i in range(M)]

    # Allocation for V and R
    V = np.zeros((D, M-1))
    R = [np.zeros((D,D)) for i in range(D)]

    while True:
        conv_ = 0.0
        for e2 in range(2):
            for i in range(M):
                for d in range(D):
                    x = 0
                    for j in range(i):
                        V[:, x] = C[i][j] @ boldU[j][:,d]
                        x += 1
                    for j in range(i+1, M):
                        V[:, x] = np.transpose(boldU[j][:, d]) @ C[j][i]
                        x += 1
                    R[d] = V @ np.transpose(V)

                L = trifact(sum(R))

                for d in range(D):
                    y = linalg.solve(np.transpose(L), linalg.solve(L,(R[d] @ boldU[i][:,d])))
                    boldU[i][:,d] = y/np.sqrt(y @ R[d] @ y)
                
                conv_ += np.sum(np.square(boldU[i]))
        conv_ = conv_ / (2*D^2*M)
        conv = 1.0 if iter_ == 1 else abs((conv_-oldconv)/oldconv)
        verbose and print("iteration: " + str(iter_) + "; convergence: " + str(conv))
        oldconv = deepcopy(conv_)
        if (0.0 <= conv <= tol):
            break
        
        if iter_ == maxiter:
            Warning("Maximum number of iterations (" + str(iter_) + ") reached before convergence!!!")            
            break

        iter_ += 1
    return boldU


def fast_alignment(boldT, boldU, Tnew):
    from scipy import linalg

    M, D = len(boldT), boldT[0].shape[0]

    # Get cross-covariances for the new domain
    C = [Tnew @ boldT[j].T for j in range(M)]

    # Memory allocation for the variables
    V = np.zeros((D,M))
    R = [np.zeros((D,D)) for d in range(D)]
    Unew = np.zeros((D,D))

    for d in range(D):
        for j in range(M):
            V[:,j] = C[j] @ boldU[j][:,d]
        R[d] = V @ V.T
    
    H = linalg.sqrtm(linalg.inv(sum(R)))

    for d in range(D):
        ev = linalg.eigh(H @ R[d] @ H.T)[1]
        Unew[:,d] = H @ ev[:,-1]
    
    return Unew


def joal_initializer(boldT, wh_type="smart"):
    M = len(boldT)
    U_init = [[] for m in range(M)]
    if wh_type == "smart":
        boldC = [[[] for i in range(j+1)] for j in range(M)]
        for j in range(M):
            for i in range(j+1, M):
                boldC[i][j] = boldT[i] @ boldT[j].T
        
        H = np.zeros((boldT[0].shape[0],boldT[0].shape[0]))
        for m in range(M):
            H = np.zeros((boldT[0].shape[0],boldT[0].shape[0]))
            for j in range(M):
                if m > j:
                    H += boldC[m][j]
                if m < j:
                    H += boldC[j][m].T
            F = scipy.linalg.svd(H)
            U_init[m] = F[0]
        
        return U_init
    else:
        return None