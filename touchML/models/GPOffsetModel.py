# -*- coding: utf-8 -*-
"""
This file is part of the TouchML toolkit.
Please visit http://www.medien.ifi.lmu.de/touchml/
@author: Daniel Buschek
"""
import Kernel
import scipy.linalg as splin
import numpy as np
from touchML.data import DBLoader


class GPOffsetModel(object):


    def __init__(self, gamma=2, noiseVar=0.001, diag=0.9, kernelMix=0.1):
        """
        Constructor for creating a new Gaussian Process offset model.

        Parameters:
        gamma - the length-scale of the Gaussian kernel (gamma in [1])
        noiseVar - the variance of the assumed noise (sigma squared in [1])
        diag - dependency of x and y (alpha in [1])
        kernelMix - the relative weight of the linear and non-linear parts of the kernel function ('a' in [1])
        """
        
        self.mX = None
        self.targetsX = None
        self.targetsY = None
         
        self.targets = None
        self.mC_chol = None
        self.mC_inv_mult_targets = None
             
        self.gamma = gamma
        self.noiseVar = noiseVar
        self.diag = diag
        
        self.kernelMix = kernelMix
        
            
        
    def fit(self, touchData, targetsAreOffsets=False):     
        """
        Fits the model to the given training data.

        Parameters:
        touchData - 2D array, each row is one touch with intended target, with columns touch x and y, and target x and y
        targetsAreOffsets - boolean, defaults to false; if true, the 3rd and 4th column of each row in the touchData array 
        are interpreted as measured offsets directly, instead of target locations
        
        Returns:
        This method has no return value.
        """
        
        inputs = np.matrix(touchData[:, 0:2])
        if not targetsAreOffsets:
            targetsX = np.matrix(touchData[:, 2] - touchData[:, 0]).T
            targetsY = np.matrix(touchData[:, 3] - touchData[:, 1]).T
        else:
             targetsX = np.matrix(touchData[:, 2]).T
             targetsY = np.matrix(touchData[:, 3]).T   
             
        self.targetsX = targetsX
        self.targetsY = targetsY
        
        'Stack targets:'
        self.targets = np.vstack((targetsX, targetsY))
        
        'Create design matrix:'
        n, m = np.shape(inputs)
        self.mX = np.matrix(inputs)

        
        'Create covariance matrix / kernel:'        
        mC = Kernel.createMixedKernel(self.mX, self.mX, self.gamma, self.kernelMix)
       
        mC_stacked_left = np.vstack((mC, mC * self.diag))
        mC_stacked_right = np.vstack((mC * self.diag, mC))
        mC_stacked = np.hstack((mC_stacked_left, mC_stacked_right))   
         
        mC_stacked_noised = mC_stacked + np.identity(2 * n) * self.noiseVar
        
        self.mC_chol = splin.cho_factor(mC_stacked_noised)[0]
        self.mC_inv_mult_targets = splin.cho_solve((self.mC_chol, False), self.targets)


    
    def predict(self, inputs, returnVar=False):
        """
        Predicts offsets for the given touches.
        Only call this method after fitting the model.
        
        Parameters:
        inputs - 2D array, each row is one touch with columns x and y
        returnVar - boolean, defaults to false; if false this method only returns the mean prediction for each touch;
        			if true, the method also returns the predictive covariance matrix for each touch
        
        Returns:
        If returnVar is set to false, this method returns a 2D array:
        each row contains the x and y mean predictions for the touch in the corresponding row of the inputs array
        
        If returnVar is set to true, this method returns two objects:
        means: the array of mean predictions as the described above
        covs: an array of 2D arrays, where the i-th entry is the predictive covariance matrix for the i-th touch in the inputs array
        """
        
        if len(np.shape(inputs)) == 1:
            inputs = np.array([inputs])
        
        inputs_N, inputs_M = np.shape(inputs)
        inputs = np.matrix(inputs)
        
        'Create and stack up covariances:'
        input_cov = Kernel.createMixedKernel(inputs, self.mX, self.gamma, self.kernelMix)
        input_cov_stacked_left = np.vstack((input_cov, input_cov * self.diag))
        input_cov_stacked_right = np.vstack((input_cov * self.diag, input_cov))
        input_cov_stacked = np.hstack((input_cov_stacked_left, input_cov_stacked_right))
        
        pred = input_cov_stacked * self.mC_inv_mult_targets
        
        'Rearrange long xy output vector to x,y (two columns)'
        predX = pred[0:len(pred) / 2]
        predY = pred[len(pred) / 2:]
        pred = np.hstack((predX, predY))
        
        if not returnVar:
            return pred
        
        # diagonal contains self covs for all input points:
        inputs_self_cov = Kernel.createMixedKernel(inputs, inputs, self.gamma, self.kernelMix)
        inputs_self_cov_stacked_left = np.vstack((inputs_self_cov, inputs_self_cov * self.diag))
        inputs_self_cov_stacked_right = np.vstack((inputs_self_cov * self.diag, inputs_self_cov))
        inputs_self_cov_stacked = np.hstack((inputs_self_cov_stacked_left, inputs_self_cov_stacked_right))
        
        mega_cov_matrix = inputs_self_cov_stacked - input_cov_stacked * splin.cho_solve((self.mC_chol, False), input_cov_stacked.T)
                
        pred_mCov_list = []
        for i in xrange(inputs_N):
            pred_mCov_i = np.zeros((2, 2))
            pred_mCov_i[0, 0] = mega_cov_matrix[i, i]
            pred_mCov_i[0, 1] = mega_cov_matrix[i, i+inputs_N]
            pred_mCov_i[1, 0] = mega_cov_matrix[i+inputs_N, i]
            pred_mCov_i[1, 1] = mega_cov_matrix[i+inputs_N, i+inputs_N] 
            pred_mCov_list.append(np.matrix(pred_mCov_i))
        
        return pred, pred_mCov_list
        
        
        
        
if __name__ == '__main__':
    
    print 'Example of using GPOffsetModel:\n\n'

    print 'Load touch data (user 1, task 1, 1st session):'
    touchData = DBLoader.loadForUserAndTask('touchesDB.sqlite', 1, 1, 0)
    print 'done'
    
    print 'Train GP model:'
    gp = GPOffsetModel(gamma=10) 
    gp.fit(touchData)
    print 'done\n\n'
    
    'Compute predictions (means, covariances) for first four training examples:'
    testData = touchData[0:4, 0:2]
    print 'Predict target locations for these touch locations (x | y):'
    print testData
    pred_means, pred_mCovs = gp.predict(testData, returnVar=True)
    print '\npredictive means, i.e. offsets (x | y):'
    print pred_means
    print '\nadd predicted offsets to touch locations to get predicted target locations:'
    print testData + pred_means
    print '\npredictive covariances (xx | xy)'
    print '                       (yx | yy)'
    print pred_mCovs
    
    
