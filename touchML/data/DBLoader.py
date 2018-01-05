# -*- coding: utf-8 -*-
"""
This file is part of the TouchML toolkit.
Please visit http://www.medien.ifi.lmu.de/touchml/
@author: Daniel Buschek
"""

import sqlite3
import os
import numpy as np


hard_outlier_threshold = 0.1


class SimpleDBHandler(object):
     
    def __init__(self, dbFile):

        self.dbFile = dbFile#os.path.join(os.path.dirname(__file__), dbFile)
        
        'Create or connect to local db (just a file):'
        self.connection = sqlite3.connect(self.dbFile)   
            
        'Fetch cursor to work with db:'
        self.cursor = self.connection.cursor()
        
    

'Filter all touches with offsets large than 3 stds of all offsets in this touch data:'
def simpleOutlierFiltering(touchData, trialMode):

    diffs = np.matrix(touchData[:, 0:2]-touchData[:, 2:4])
    diffs_squared = np.power(diffs, 2)  # element-wise squared
    diffs_summed = diffs_squared * np.ones((2, 1))  # sum of rows
    diffs_sum_root = np.sqrt(diffs_summed)  # square root
    
    std = np.std(diffs_sum_root, 0)
    mean = np.average(diffs_sum_root, 0)
    
    touchData_clean = []
    outliers = []
    outlierIndices = []
    i = 0
    for t in touchData:
        d = (t[0:2] - t[2:4])   
        dist = np.sqrt(np.dot(d,d))
        if dist - mean > 3 * std or dist > hard_outlier_threshold:
            outliers.append(t)
            outlierIndices.append(i)
        else:
            touchData_clean.append(t)
        i += 1
    touchData_clean = np.array(touchData_clean)
    outliers = np.array(outliers)
    
    return touchData_clean, outliers, outlierIndices



def loadData(dbFile, trialID, normalise=True, filterOutliers=True, returnTrialData=False):
    
    db = SimpleDBHandler(dbFile)
    db.cursor.execute("SELECT t.touchUpX, t.touchUpY, t.targetX, t.targetY FROM trials tr, taps t WHERE t.trialID = tr.id AND tr.id = '%s'" % (trialID))
    
    touchData = []
    for row in db.cursor:
        touchData.append([float(v) for v in row])      
    touchData = np.array(touchData)
    
    
    db.cursor.execute("SELECT screenW, screenH, trialMode, sessionIndex FROM trials WHERE id = '%s'" % (trialID))
    trialData = db.cursor
    for row in db.cursor:
        trialData = row
    trialData = np.array(trialData)
    
    if normalise:
        touchData[:, 0] /= float(trialData[0])
        touchData[:, 2] /= float(trialData[0])
        touchData[:, 1] /= float(trialData[1])
        touchData[:, 3] /= float(trialData[1])
        
        
    'Outlier filtering:'
    if filterOutliers:
        touchData, outliers, outlierIndices = simpleOutlierFiltering(touchData, trialData[2])    
        
    db.cursor.close()
    if returnTrialData:
        return touchData, trialData
    else:
        return touchData




def loadForUserAndTask(dbFile, userID, task, session, normalise=True, filterOutliers=True, returnTrialData=False):
    
    db = SimpleDBHandler(dbFile)
    db.cursor.execute("SELECT id FROM trials WHERE subjectID = '%s' AND trialMode='%s' AND sessionIndex='%s'" % (userID, task, session))
    trialID = -1
    for row in db.cursor:
        trialID = row[0]
    db.cursor.close()
    return loadData(dbFile, trialID, normalise, filterOutliers, returnTrialData)
    
    
    
if __name__ == '__main__':
    
    print 'If you get an error below, make sure to copy ' \
    + 'the database file "touchesDB.sqlite" from TouchML\'s "Dataset" folder ' \
    + 'to the "data" folder of the Python library!'
    touchData, trialData = loadForUserAndTask('touchesDB.sqlite', 1, 1, 0, returnTrialData=True)
    print trialData    
    print np.shape(touchData)
    
