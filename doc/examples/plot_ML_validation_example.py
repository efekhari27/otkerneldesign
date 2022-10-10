"""
Machine learning validation examples
====================================

The aim of this page is to provide simple use-case where kernel herding 
is used to build a design of experiments complementary to an existing one, 
either to enhance a machine learning model or for validation.
"""
import numpy as np
import openturns as ot
import otkerneldesign as otkd
import matplotlib.pyplot as plt
from matplotlib import cm

# %%
# The following helper class will make plotting easier.

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

    def draw_2D_contour(self, title, function=None, distribution=None, colorbar=cm.coolwarm):
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
        plt.title(title)
        plt.xlabel("$x_0$")
        plt.ylabel("$x_1$")
        return fig

# %%
# Building a regression model of a 2D function
# --------------------------------------------


