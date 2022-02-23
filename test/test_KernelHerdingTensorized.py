# -*- coding: utf-8 -*-
"""
Test for KernelHerdingTensorized class.
"""
import otkerneldesign as otkd
import unittest
import openturns as ot
import openturns.testing as ott


class CheckKernelHerdingKernelHerdingTensorized(unittest.TestCase):
    def test_KernelHerding(self):
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
        uniform_design_points, _ = kht.select_design(size)

        ref = [[0.5,0.5],[0.222412,0.781494],[0.760498,0.229736],[0.244141,0.244141],[0.751953,0.751953],[0.891357,0.485596],[0.122559,0.505371],[0.492676,0.869629],[0.496582,0.131348],[0.908203,0.908203],[0.354736,0.635498],[0.092041,0.0993652],[0.630859,0.373047],[0.908447,0.095459],[0.0917969,0.912109],[0.358643,0.365967],[0.626953,0.626953],[0.664307,0.948975],[0.926025,0.671631],[0.328369,0.0505371]]
        
        ott.assert_almost_equal(uniform_design_points, ref, 1e-4, 0.0)


if __name__ == "__main__":
    unittest.main()