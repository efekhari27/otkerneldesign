# -*- coding: utf-8 -*-
"""
Test for KernelHerdingTensorized class.
"""
import otkerneldesign as otkd
import unittest
import openturns as ot
import openturns.testing as ott


class CheckKernelHerdingTensorized(unittest.TestCase):
    def test_KernelHerdingTensorized(self):
        size = 20
        dimension = 2
        distribution = ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * dimension)
        ker_list = [ot.MaternModel([0.1], [1.0], 2.5)] * dimension
        kernel = ot.ProductCovarianceModel(ker_list)

        kht = otkd.KernelHerdingTensorized(
            kernel=kernel,
            candidate_set_size=2 ** 12,
            distribution=distribution
        )
        uniform_design_points = kht.select_design(size)
        smaller_design = kht.select_design(size // 2)

        ref = [[0.5, 0.5] ,[0.78125, 0.78125], [0.240967, 0.233643], [0.237061, 0.768799], [0.755859, 0.248047], [0.106201, 0.489502], [0.501465, 0.876465], [0.873535, 0.504395], [0.500488, 0.127441], [0.912109, 0.0917969], [0.639404, 0.647705], [0.0878906, 0.0878906], [0.35376, 0.384521], [0.095459, 0.908447], [0.362061, 0.643799], [0.908203, 0.908203], [0.630859, 0.373047], [0.67749, 0.045166], [0.952881, 0.332275], [0.0705566, 0.667725]]

        ott.assert_almost_equal(uniform_design_points, ref, 1e-4, 0.0)
        ott.assert_almost_equal(smaller_design, ref[:size // 2], 1e-4, 0.0)


if __name__ == "__main__":
    unittest.main()