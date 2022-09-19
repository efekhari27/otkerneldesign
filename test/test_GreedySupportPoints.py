# -*- coding: utf-8 -*-
"""
Test for GreedySupportPoints class.
"""
import otkerneldesign as otkd
import unittest
import openturns as ot
import openturns.testing as ott


class CheckGreedySupportPoints(unittest.TestCase):
    def test_GreedySupportPoints(self):
        size = 20
        dimension = 2
        distribution = ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * dimension)

        sp = otkd.GreedySupportPoints(
            candidate_set_size=2 ** 12,
            distribution=distribution
        )
        uniform_design_points = sp.select_design(size)
        smaller_design = sp.select_design(size // 2)

        ref = [[0.5, 0.5], [0.28125, 0.28125], [0.793701171875, 0.802001953125], [0.712890625, 0.212890625], [0.181640625, 0.806640625], [0.83642578125, 0.46337890625], [0.076904296875, 0.460205078125], [0.51171875, 0.83203125], [0.46484375, 0.03515625], [0.70166015625, 0.64501953125], [0.26171875, 0.58203125], [0.89892578125, 0.15087890625], [0.26318359375, 0.95849609375], [0.17578125, 0.10546875], [0.9453125, 0.7265625], [0.48046875, 0.30078125], [0.390625, 0.671875],  [0.69775390625, 0.35205078125], [0.0576171875, 0.6806640625], [0.8154296875, 0.9384765625]]

        ott.assert_almost_equal(uniform_design_points, ref, 1e-4, 0.0)
        ott.assert_almost_equal(smaller_design, ref[:size // 2], 1e-4, 0.0)


if __name__ == "__main__":
    unittest.main()