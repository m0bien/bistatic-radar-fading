import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry import Position3D, calculate_bistatic_geometry
from src.reflection import (
    GroundMedium,
    Polarization,
    fresnel_reflection_coefficient,
    surface_roughness_factor
)
from src.propagation import BistaticTwoRayModel
from src.radar_equation import bistatic_rcs_model, calculate_bistatic_received_power
from src.diversity import DiversityCombiner, FrequencyDiversityAnalyzer, SpatialDiversityAnalyzer

class TestBistaticRadarTwoRay(unittest.TestCase):
    def setUp(self):
        self.tx = Position3D(x=-10000.0, y=0.0, z=20.0)
        self.rx = Position3D(x=10000.0, y=0.0, z=10.0)
        self.tgt = Position3D(x=0.0, y=20000.0, z=1000.0)

    def test_geometry_path_lengths(self):
        geo = calculate_bistatic_geometry(self.tx, self.rx, self.tgt)
        # Direct paths must be shorter than reflected paths
        self.assertGreater(geo["r_tx_refl"], geo["r_tx_direct"])
        self.assertGreater(geo["r_rx_refl"], geo["r_rx_direct"])
        self.assertGreater(geo["delta_r_tx"], 0.0)
        self.assertGreater(geo["delta_r_rx"], 0.0)
        # Grazing angles must be positive
        self.assertGreater(geo["psi_tx_rad"], 0.0)
        self.assertGreater(geo["psi_rx_rad"], 0.0)
        # Bistatic angle must be between 0 and 180 deg
        self.assertTrue(0.0 <= geo["bistatic_angle_deg"] <= 180.0)

    def test_fresnel_reflection(self):
        psi = np.radians(10.0)
        gamma_pec_h = fresnel_reflection_coefficient(psi, 3e9, Polarization.HORIZONTAL, GroundMedium.PERFECT_CONDUCTOR)
        gamma_pec_v = fresnel_reflection_coefficient(psi, 3e9, Polarization.VERTICAL, GroundMedium.PERFECT_CONDUCTOR)
        self.assertAlmostEqual(gamma_pec_h, -1.0 + 0j)
        self.assertAlmostEqual(gamma_pec_v, 1.0 + 0j)

        # For medium ground, |Gamma| <= 1.0
        gamma_med = fresnel_reflection_coefficient(psi, 3e9, Polarization.HORIZONTAL, GroundMedium.MEDIUM_GROUND)
        self.assertLessEqual(np.abs(gamma_med), 1.0)

    def test_two_ray_propagation_bounds(self):
        model = BistaticTwoRayModel(frequency_hz=3e9, polarization=Polarization.HORIZONTAL)
        res = model.compute(self.tx, self.rx, self.tgt)
        # For a single link |F_t| <= 1 + |Gamma| <= 2.0 -> |F_t|^2 <= 4.0
        self.assertLessEqual(res["f_tx_sq"], 4.01)
        self.assertLessEqual(res["f_rx_sq"], 4.01)
        # Total F^4 <= 16.0 (12.04 dB)
        self.assertLessEqual(res["f_total_pwr"], 16.05)

    def test_diversity_combiners(self):
        # 3 branches, 5 samples
        branches = np.array([
            [1.0, 10.0, 0.1, 5.0, 2.0],
            [4.0, 2.0, 8.0, 0.5, 3.0],
            [2.0, 5.0, 4.0, 12.0, 1.0]
        ])
        sc = DiversityCombiner.selection_combining(branches)
        mrc = DiversityCombiner.maximal_ratio_combining(branches)
        egc = DiversityCombiner.equal_gain_combining(branches)

        np.testing.assert_array_almost_equal(sc, [4.0, 10.0, 8.0, 12.0, 3.0])
        np.testing.assert_array_almost_equal(mrc, [7.0, 17.0, 12.1, 17.5, 6.0])
        # MRC must always be >= EGC >= SC for positive SNRs
        self.assertTrue(np.all(mrc >= sc))
        self.assertTrue(np.all(mrc >= egc))

if __name__ == "__main__":
    unittest.main()
