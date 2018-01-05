# -*- coding: utf-8 -*-
"""
This file is part of the TouchML toolkit.
Please visit http://www.medien.ifi.lmu.de/touchml/
@author: Daniel Buschek
"""
import numpy as np

class LinearModel(object):
    """
    A linear regression model, used by LinearOffsetModel.py.
    Most users will not use this class directly, but rather create a LinearOffsetModel object.
    """
    

    def __init__(self, lambda_reg=0):
        
        self.mX = None
        self.y = None
        self.w = None
        self.mInv = None
        self.noiseVar = None
        self.lamba_reg = lambda_reg
        
        
    def fit(self, mX, y):
        
        n, m = np.shape(mX)
        self.mX = mX
        self.y = y
        
        self.mInv = np.linalg.inv(self.mX.T * self.mX + self.lamba_reg * np.identity(m))
        self.w = self.mInv * self.mX.T * y
        self.w = self.w.T
        
        self.noiseVar = float(1. / n * (np.dot(y.T, y) - y.T * mX * self.w.T))
        
        
        
    def predict(self, x):
        
        pred = np.dot(self.w, x)
        return pred
    
    
    def predictWithVar(self, x):
    
        pred = np.dot(self.w, x)         
        predVar = self.noiseVar * x.T * self.mInv * x       
        return [float(pred), float(predVar)]
