# -*- coding: utf-8 -*-
"""
This file is part of the TouchML toolkit.
Please visit http://www.medien.ifi.lmu.de/touchml/
@author: Daniel Buschek
"""
import numpy as np

def createRBFKernel(M, M2, gamma):
    
    n, m = np.shape(M)
    n2, m2 = np.shape(M2)
    
    M_sq = np.multiply(M, M)
    M2_sq = np.multiply(M2, M2)
    sum_M = np.matrix([np.sum(ri) for ri in M_sq]).T
    sum_M2 = np.matrix([np.sum(ri) for ri in M2_sq]).T
    
    ones_M = np.matrix(np.ones((n, 1)))
    ones_M2 = np.matrix(np.ones((1, n2)))
       
    distance = sum_M * ones_M2 + ones_M * sum_M2.T - 2 * M * M2.T  
    K = np.exp(-gamma * distance)        
    return K


def createLinearKernel(M, M2):
    
    K = M * M2.T
    return K

def createMixedKernel(M, M2, gamma, mixAlpha):
    
    K1 = createRBFKernel(M, M2, gamma)
    K2 = createLinearKernel(M, M2)
    K = mixAlpha * K2 + (1-mixAlpha)*K1
    return K

def kernelFunctionRBF(a, b, gamma):
    
    diff = a - b
    return np.exp(-gamma * np.dot(diff, diff.T))