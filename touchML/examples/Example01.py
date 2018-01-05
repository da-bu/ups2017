# -*- coding: utf-8 -*-
"""
This file is part of the TouchML toolkit.
Please visit http://www.medien.ifi.lmu.de/touchml/
@author: Daniel Buschek
"""
from models.GPOffsetModel import GPOffsetModel
from models.LinearOffsetModel import LinearOffsetModel
import numpy as np


data = np.array([[.1,.12,.21,.2],[.32,.3,.42,.4],[.53,.5,.75,.7],
                 [.6,.65,.8,.8],[.77,.7,.73,.73],[.93,.91,.99,.99],
                 [.1,.3,.2,.4]]);


gp = GPOffsetModel(gamma=2, noiseVar=0.001, diag=0, kernelMix=0.9)
gp.fit(data)

print gp.predict([[.32,.3]], True)


lm = LinearOffsetModel()
lm.fit(data)

print lm.predict([[.32,.3]], True)


print 'See more examples by running each model file (GPOffsetModel.py, LinearOffsetModel.py)'

