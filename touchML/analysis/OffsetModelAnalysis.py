# -*- coding: utf-8 -*-
"""
This file is part of the TouchML toolkit.
Please visit http://www.medien.ifi.lmu.de/touchml/
@author: Daniel Buschek
"""
import numpy as np




def createFolds(data, numFolds):
    """
    Splits the given data into the given number of folds (i.e. parts). 
    Note that this is NOT a random split. If you need a random split, you have to shuffle
    the data yourself before calling this method.

    Parameters:
    data - 2D array, each row is one touch with columns touch x and y, and target x and y
    numFolds - number of folds

    Returns:
    A list of length numFolds; the i-th list entry is i-th part of the given dataset.
    """
    
    N = len(data)
    tuplesPerFold = int(np.floor(N / numFolds))
    
    dataAll = []
    for i in xrange(1, numFolds + 1):
        dataFold = data[(i - 1) * tuplesPerFold:i * tuplesPerFold]
        dataAll.append(dataFold)
    
    return dataAll



def computeRMSEfromErrors(errors):
    """
    Computes the Root-Mean-Square-Error (RMSE) for the given set of errors 
    (i.e. errors = euclidean touch to target distances = touch offsets).

    Parameters:
    errors - a list of touch-target-distances

    Returns:
    The RMSE of the given list of errors.
    """
    
    errors = np.array(errors)
    n = len(errors)
    rmse = np.sqrt(np.sum(errors ** 2) / n)
    return rmse
    
    

    
    
def computeImprovementInRMSE(touchData, offsetModel, featureIndices=(0,1)):
    """
    Computes the improvement in RMSE achieved with the given trained model
    applied to the given data (i.e. the test set), compared to the RMSE without
    the models predictions.

    Parameters:
    touchData - 2D array, each row is one touch with columns touch x and y, and target x and y
    offsetModel - a trained offset model

    Returns:
    The improvement in RMSE of the touch offsets in percent
    """
      
    preds = offsetModel.predict(touchData[:,featureIndices])
             
    'Compute euclidean distance of predictions and targets:'
    diffs = touchData[:,2:4] - (touchData[:,0:2]+preds)  # difference
    diffs_squared = np.power(diffs, 2)  # element-wise squared
    diffs_summed = diffs_squared * np.ones((2, 1))  # sum of rows
    diffs_sum_root = np.sqrt(diffs_summed)  # square root
    
    
    diffs2 = np.matrix(touchData[:,2:4] - touchData[:,0:2])  # difference
    diffs_squared2 = np.power(diffs2, 2)  # element-wise squared
    diffs_summed2 = diffs_squared2 * np.ones((2, 1))  # sum of rows
    diffs_sum_root2 = np.sqrt(diffs_summed2)  # square root
    
    rmse_with_model = computeRMSEfromErrors(diffs_sum_root)
    rmse_no_model = computeRMSEfromErrors(diffs_sum_root2)

    diff = 100*(rmse_no_model - rmse_with_model)/rmse_no_model
    # print rmse_no_model, rmse_with_model, diff
    return diff
    
    




def computeRMSE(touchData, offsetModel):
    """
    Computes RMSE achieved with the given trained model applied to 
    the given data (i.e. the test set).

    Parameters:
    touchData - 2D array, each row is one touch with columns touch x and y, and target x and y
    offsetModel - a trained offset model

    Returns:
    The RMSE of the touch offsets when corrected with the model.
    """
      
    preds = offsetModel.predict(touchData[:,0:2])
    
    'Compute euclidean distance of predictions and targets:'
    diffs = touchData[:,2:4] - (touchData[:,0:2]+preds)  # difference
    diffs_squared = np.power(diffs, 2)  # element-wise squared
    diffs_summed = diffs_squared * np.ones((2, 1))  # sum of rows
    diffs_sum_root = np.sqrt(diffs_summed)  # square root
    
    rmse_with_model = computeRMSEfromErrors(diffs_sum_root)
    return rmse_with_model


def crossValidationRMSE(offsetModel, touchData, numFolds, targetsAreOffsets=False, returnMean=True):
    """
    Computes RMSE with the given (untrained) model applied to the given data 
    when evaluated with cross-validation with the given number of folds.

    Parameters:
    offsetModel - a trained offset model
    touchData - 2D array, each row is one touch with columns touch x and y, and target x and y
    numFolds - number of folds for the cross validation
    targetsAreOffsets - boolean, defaults to false; 
        if true, the 3rd and 4th column of each row in the touchData array
        are interpreted as measured offsets directly, instead of target locations
    returnMean - boolean, defaults to true;
        if true, this method returns the mean RMSE over all folds,
        if false, the list of RMSEs for all folds is returned instead
    
    Returns:
    If returnMean is true: the mean RMSE of the touch offsets 
        when corrected with the model, as evaluated with cross-validation.
    If returnMean is false: a list of length numFolds, containing the RMSEs for all folds.
    """
    
    rmse_sum = 0   
    'split data into folds:'
    data_folded = createFolds(touchData, numFolds)
    rmses = []
    
    'for each fold...'
    for fold in xrange(numFolds):
        
        'create train and test sets:'
        trainingData = np.vstack([data_folded[i] for i in xrange(numFolds) if i != fold])
        testData = data_folded[fold] 
        
        'train model:'
        offsetModel.fit(trainingData, targetsAreOffsets)
        
        'compute and sum up improvement:'
        rmse = computeRMSE(testData, offsetModel)
        #print fold, improvement
        rmse_sum += rmse
        rmses.append(rmse)
        
    'average and return:'
    mean_rmse = rmse_sum / numFolds
    return mean_rmse if returnMean else rmses



def crossValidationImprovementRMSE(offsetModel, touchData, numFolds, targetsAreOffsets=False, returnMean=True):
    """
    Computes the improvement in RMSE achieved with the given (untrained) model applied to 
    the given data when evaluated with cross-validation with the given number of folds.
    
    Parameters:
    offsetModel - a trained offset model
    touchData - 2D array, each row is one touch with columns touch x and y, and target x and y
    numFolds - number of folds for the cross validation
    targetsAreOffsets - boolean, defaults to false; 
        if true, the 3rd and 4th column of each row in the touchData array
        are interpreted as measured offsets directly, instead of target locations
    returnMean - boolean, defaults to true;
        if true, this method returns the mean improvement in RMSE over all folds,
        if false, the list of improvements in RMSE for all folds is returned instead
    
    Returns:
    If returnMean is true: the mean improvement in RMSE of the touch offsets 
        when corrected with the model, as evaluated with cross-validation.
    If returnMean is false: a list of length numFolds, containing the improvements in RMSE for all folds.
    """
    
    improvement_sum = 0   
    'split data into folds:'
    data_folded = createFolds(touchData, numFolds)
    improvements = []
    
    'for each fold...'
    for fold in xrange(numFolds):
        
        'create train and test sets:'
        trainingData = np.vstack([data_folded[i] for i in xrange(numFolds) if i != fold])
        testData = data_folded[fold]  
        
        'train model:'
        offsetModel.fit(trainingData, targetsAreOffsets)
        
        'compute and sum up improvement:'
        improvement = computeImprovementInRMSE(testData, offsetModel)
        #print fold, improvement
        improvement_sum += improvement
        improvements.append(improvement)
        
    'average and return:'
    mean_improvement = improvement_sum / numFolds
    return mean_improvement if returnMean else improvements
