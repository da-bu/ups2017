# -*- coding: utf-8 -*-
"""
This file is part of the TouchML toolkit.
Please visit http://www.medien.ifi.lmu.de/touchml/
@author: Daniel Buschek
"""
import csv
import numpy as np 


def loadTouchData(filename, delimiter=","):

    csvfile = open(filename)
    csv_reader = csv.reader(csvfile, delimiter=delimiter)
    touchData = np.array([row for row in csv_reader])
    csvfile.close()
    
    return touchData