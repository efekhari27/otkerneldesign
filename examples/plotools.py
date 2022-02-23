# -*- coding: utf-8 -*-
"""
@authors: E.Fekhari
File description TO DO 
"""

import numpy as np
import openturns as ot
from matplotlib import cm
import matplotlib.pyplot as plt

###########################################
############ FUNCTIONS VISUALS ############
###########################################
#my_colorbar = cm.Spectral.reversed() #Nice different colorbar
class DrawFunctions:
    def __init__(self):
        dim = 2
        self.grid_size = 100
        lowerbound = [0.] * dim
        upperbound = [1.] * dim
        mesher = ot.IntervalMesher([self.grid_size-1] * dim)
        interval = ot.Interval(lowerbound, upperbound)
        mesh = mesher.build(interval)
        self.nodes = mesh.getVertices()
        self.X0, self.X1 = np.array(self.nodes).T.reshape(2, self.grid_size, self.grid_size)

    def draw_2D_controur(self, title, function=None, distribution=None, colorbar=cm.coolwarm):
        fig = plt.figure(figsize=(7, 6))
        if distribution is not None:
            Zpdf = np.array(distribution.computePDF(self.nodes)).reshape(self.grid_size, self.grid_size)
            nb_isocurves = 9
            contours = plt.contour(self.X0, self.X1, Zpdf, nb_isocurves, colors='black', alpha=0.6)
            plt.clabel(contours, inline=True, fontsize=8)
        if function is not None:
            Z = np.array(function(self.nodes)).reshape(self.grid_size, self.grid_size)
            plt.contourf(self.X0, self.X1, Z, 18, cmap=colorbar)
            plt.colorbar()
        plt.title(title, fontsize=16)
        plt.xlabel("$x_0$", fontsize=14)
        plt.ylabel("$x_1$", fontsize=14)
        #plt.close()
        return fig