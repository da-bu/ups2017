# -*- coding: utf-8 -*-
"""
This file is part of the TouchML toolkit.
Please visit http://www.medien.ifi.lmu.de/touchml/
@author: Daniel Buschek
"""
import matplotlib.pyplot as plt
import numpy as np
from touchML.data import DBLoader



def plot(touchData, w=1, h=1):
    """
    Creates a scatter plot of the given target locations.
    Targets are connected to their corresponding touches by lines.
    
    Parameters:
    touchData - the touch and target locations
    w - the width of the plot, use this and 'h' to recreate your screen's aspect ratio
    h - the height of the plot, use this and 'w' to recreate your screen's aspect ratio
    """
    
    n, m = np.shape(touchData)

    plt.figure()
    plt.subplot(111, aspect=16./9)
    plt.xlim((0, w))
    plt.ylim((h,0))
        
    for i in range(n):
        
        xVals = (touchData[i, 2], touchData[i, 0])
        yVals = (touchData[i, 3], touchData[i, 1])         
        plt.plot(xVals, yVals, color='k', lw=1)
        plt.scatter(touchData[i, 2], touchData[i, 3], color='k', s=4)
    
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    
    touchData = DBLoader.loadForUserAndTask('touchesDB.sqlite', 6, 1, 0)
    plot(touchData)