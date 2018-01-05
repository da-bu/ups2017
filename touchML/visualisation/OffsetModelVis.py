# -*- coding: utf-8 -*-
"""
The original version of this file is part of the TouchML toolkit.
Please visit http://www.medien.ifi.lmu.de/touchml/
@author: Daniel Buschek
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from touchML.models.GPOffsetModel import GPOffsetModel
from touchML.models.LinearOffsetModel import LinearOffsetModel
from touchML.data import DBLoader

    
    
def arrowPlot(models, w=1, h=1, stepsX = 20, stepsY = 20, figsize=(7,7)):
    """
    Creates an arrow plot to visualise the prediction surface of the given offset model.
    
    Parameters:
    model - the offset model to visualise
    w - the width of the plot, use this and 'h' to recreate your screen's aspect ratio
    h - the height of the plot, use this and 'w' to recreate your screen's aspect ratio
    stepsX - number of arrows per row
    stepsY - number of rows
    """
     
    plt.figure(figsize=figsize)
    for m_i, model in enumerate(models):
        ax = plt.subplot(1,len(models),m_i+1, aspect=16./9)
        plt.title('User %d' % (m_i+1), fontsize=20)
        plt.xlim((0, w))
        plt.ylim((h, 0))

        'Create grid inputs:'
        xi = np.linspace(0, 1, stepsX, True)
        yi = np.linspace(0, 1, stepsY, True)
        inputs_grid = []

        for i in xrange(stepsX):
            for j in xrange(stepsY):
                inputs_grid.append([xi[i], yi[j]])
        inputs_grid = np.matrix(inputs_grid)

        'Predict batch for all grid inputs at once:'
        preds = model.predict(inputs_grid)

        'Draw grid predictions as arrows (= model surface):'
        for i in xrange(stepsX * stepsY):
            pX = inputs_grid[i, 0]
            pY = inputs_grid[i, 1]
            predX = preds[i, 0]
            predY = preds[i, 1]

            if predX != 0 or predY != 0:
                ax.arrow(pX, pY, predX, predY, head_width=0.02, head_length=0.015, fc='k', ec='k', lw=0.5, alpha=0.75)

    plt.tight_layout()
    plt.show()



def contourPlot(models, dim="x", w=1, h=1, stepsX=20, stepsY=20, figsize=(9,9), hide_colorbar=False):
    """
    Creates a contour plot to visualise one dimension of the prediction surface of the given offset model.
    
    Parameters:
    model - the offset model to visualise
    dim - either 'x' or 'y', sets the dimension to visualise
    w - the width of the plot, use this and 'h' to recreate your screen's aspect ratio
    h - the height of the plot, use this and 'w' to recreate your screen's aspect ratio
    stepsX - number of arrows per row
    stepsY - number of rows
    """
    
    if dim=="x":
        dim = 0
    elif dim=="y":
        dim = 1
        
    plt.figure(figsize=figsize)
    for m_i, model in enumerate(models):
        ax = plt.subplot(1, len(models), m_i + 1, aspect=16. / 9)
        plt.title('User %d' % (m_i + 1), fontsize=20)
        plt.xlim((0, w))
        plt.ylim((h, 0))

        'Create grid inputs:'
        xi = np.linspace(0, 1, stepsX, True)
        yi = np.linspace(0, 1, stepsY, True)
        inputs_grid = []

        for i in xrange(stepsY):
            for j in xrange(stepsX):
                inputs_grid.append([xi[j], yi[i]])
        inputs_grid = np.matrix(inputs_grid)


        'Predict batch for all grid inputs at once:'
        preds = model.predict(inputs_grid)

        preds2D = np.zeros((stepsY, stepsX, 2))
        for i in xrange(stepsY):
            for j in xrange(stepsX):
                pred = preds[i*stepsX+j]
                preds2D[i, j] = pred

        levs = np.linspace(-np.max(np.abs(preds2D)), np.max(np.abs(preds2D)), 16)
        CS = plt.contourf(xi, yi, preds2D[:, :, dim], levels=levs, cmap=plt.cm.BrBG)
        CS2 = plt.contour(xi, yi, preds2D[:, :, dim], levels=levs, colors="k")
        if not hide_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=-0.5)
            plt.colorbar(CS, cax=cax, ticks=[-0.1, -0.05, 0, 0.05, 0.1])

    plt.tight_layout()
    plt.show()



def variancePlot(models, titles=['User 1', 'User 2'], w=1, h=1, stepsX=20, stepsY=20, figsize=(9, 9), hide_colorbar=False):
    """
    Creates a contour plot to visualise the predictive variance of the given offset model.
    
    Parameters:
    model - the offset model to visualise
    w - the width of the plot, use this and 'h' to recreate your screen's aspect ratio
    h - the height of the plot, use this and 'w' to recreate your screen's aspect ratio
    stepsX - number of arrows per row
    stepsY - number of rows
    """

    plt.figure(figsize=figsize)
    for m_i, model in enumerate(models):
        ax = plt.subplot(1, len(models), m_i + 1, aspect=16. / 9)
        #plt.title('User %d' % (m_i + 1), fontsize=20)
        if titles is not None:
            plt.title(titles[m_i], fontsize=20)
        plt.xlim((0, w))
        plt.ylim((h, 0))

        xi = np.linspace(0, 1, stepsX, True)
        yi = np.linspace(0, 1, stepsY, True)
        inputs_grid = []
        for i in xrange(stepsY):
            for j in xrange(stepsX):
                inputs_grid.append([xi[j], yi[i]])
        inputs_grid = np.matrix(inputs_grid)

        preds, mCovs = model.predict(inputs_grid, returnVar=True)

        dets = np.zeros((stepsY, stepsX))
        for i in xrange(stepsX):
            for j in xrange(stepsY):
                dets[j, i] = np.linalg.det(mCovs[j*stepsX+i])

        dets = np.log(dets)
        levs = np.linspace(np.min(dets), np.max(dets), 16)
        CS = plt.contourf(xi, yi, dets, levels=levs, cmap=plt.cm.jet, alpha=0.5)
        if not hide_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=-0.5)
            plt.colorbar(CS, cax=cax)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    
    touchData = DBLoader.loadForUserAndTask('../data/touchesDB.sqlite', 4, 1, 0)
    
    om = GPOffsetModel(gamma=2, diag=0.9, kernelMix=0.1)#LinearOffsetModel(lambda_reg=0.001)#
    om.fit(touchData)
   
    contourPlot([om], dim="x")
    #arrowPlot(om)
    #variancePlot(om)
    
