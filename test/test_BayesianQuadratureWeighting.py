# -*- coding: utf-8 -*-
"""
Test for QuadratureWeighting class.
"""
import otkerneldesign as otkd
import unittest
import openturns as ot
import openturns.testing as ott


class CheckBayesianQuadratureWeighting(unittest.TestCase):
    def test_BayesianQuadratureWeighting(self):
        size = 20
        dimension = 2
        distribution = ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * dimension)
        ot.RandomGenerator_SetSeed(1234)
        sample = distribution.getSample(size)

        ker_list = [ot.MaternModel([0.1], [1.0], 2.5)] * dimension
        kernel = ot.ProductCovarianceModel(ker_list)

        qw = otkd.BayesianQuadratureWeighting(
            kernel=kernel,
            distribution_sample_size=2 ** 12,
            distribution=distribution
        )
        optimal_weights = qw.compute_bayesian_quadrature_weights(sample)

        ref = [0.03309285, 0.04768146, 0.01494904, 0.02490442, 0.02028773, 0.0379315, 0.03331086, 0.02565772, 0.04326114, 0.01338789, 0.02757236, 0.04702677, 0.04320613, 0.03731281,-0.00650539, 0.02138246, 0.02888203, 0.0283134, 0.01463362, 0.03812451]

        ott.assert_almost_equal(optimal_weights, ref, 1e-4, 0.0)


if __name__ == "__main__":
    unittest.main()
