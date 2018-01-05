# -*- coding: utf-8 -*-
"""
This file is part of the TouchML toolkit.
Please visit http://www.medien.ifi.lmu.de/touchml/
@author: Daniel Buschek
"""
from LinearModel import LinearModel
import numpy as np
from touchML.data import DBLoader


class LinearOffsetModel(object):
    """
    This class implements regularised linear regression for 2D touch offset modelling.
    For more details, see: http://www.medien.ifi.lmu.de/touchml/

    Methods:
     - LinearOffsetModel: constructor
     - predict: predicts offsets for touch locations
     - fit: trains the model on the given touch and target locations
     """


    def __init__(self, lambda_reg=0.001):
        """
        Constructor for creating a new linear offset model.

        Parameters:
        lambdaReg - the regularisation parameter
        """

        self.mX = None
        self.targetsX = None
        self.targetsY = None
        self.lmX = None
        self.lmY = None
        self.lambda_reg = lambda_reg



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

        self.mX = np.matrix([self.applyBasisFunctions(x.T) for x in inputs])

        self.targetsX = targetsX
        self.targetsY = targetsY

        self.lmX = LinearModel(self.lambda_reg)
        self.lmX.fit(self.mX, self.targetsX)

        self.lmY = LinearModel(self.lambda_reg)
        self.lmY.fit(self.mX, self.targetsY)



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

        If returnVar is set to true, this method returns an object with two attributes:
        means: the array of mean predictions as the described above
        covs: an array of 2D arrays, where the i-th entry is the predictive covariance matrix for the i-th touch in the inputs array
        """

        if len(np.shape(inputs)) == 1:
            inputs = np.array([inputs])

        preds = []
        mCovs = []
        for x in inputs:
            x = np.matrix(self.applyBasisFunctions(x)).T
            predX, predVarX = self.lmX.predictWithVar(x)
            predY, predVarY = self.lmY.predictWithVar(x)
            predVar = np.matrix([[predVarX, 0], [0, predVarY]])
            preds.append([predX, predY])
            mCovs.append(predVar)

        preds = np.matrix(preds)
        if returnVar:
            return preds, mCovs
        else:
            return preds


    def applyBasisFunctions(self, x):
        x = np.array(x).flatten()
        return [1, x[0], x[1], x[0] ** 2, x[1] ** 2]



if __name__ == '__main__':

    print 'Example of using LinearOffsetModel:\n\n'

    print 'Load touch data (user 1, task 1, 1st session):'
    touchData = DBLoader.loadForUserAndTask('touchesDB.sqlite', 1, 1, 0)
    print 'done'

    print 'Train linear model:'
    gp = LinearOffsetModel(lambda_reg=0.1)
    gp.fit(touchData)
    print 'done\n\n'

    'Compute predictions (means, covariances) for first four training examples:'
    testData = touchData[0:4, 0:2]
    print 'predict target locations for these touch locations (x | y):'
    print testData
    pred_means, pred_mCovs = gp.predict(testData, returnVar=True)
    print '\npredictive means, i.e. offsets (x | y):'
    print pred_means
    print '\nadd predicted offsets to touch locations to get predicted target locations:'
    print testData + pred_means
    print '\npredictive covariances (xx | xy)'
    print '                       (yx | yy)'
    print pred_mCovs