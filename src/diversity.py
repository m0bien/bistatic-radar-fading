import numpy as np
from typing import List, Dict, Any
from .geometry import Position3D
from .propagation import BistaticTwoRayModel
from .radar_equation import calculate_bistatic_received_power

class DiversityCombiner:
    @staticmethod
    def selection_combining(branch_snr_lin: np.ndarray) -> np.ndarray:
        return np.max(branch_snr_lin, axis=0)

    @staticmethod
    def maximal_ratio_combining(branch_snr_lin: np.ndarray) -> np.ndarray:
        return np.sum(branch_snr_lin, axis=0)

    @staticmethod
    def equal_gain_combining(branch_snr_lin: np.ndarray) -> np.ndarray:
        m = branch_snr_lin.shape[0]
        sum_sqrt = np.sum(np.sqrt(np.maximum(branch_snr_lin, 0.0)), axis=0)
        return (sum_sqrt**2) / m

class FrequencyDiversityAnalyzer:
    def __init__(self, two_ray_model: BistaticTwoRayModel):
        self.model = two_ray_model

    def evaluate_frequency_grid(
        self,
        tx_pos: Position3D,
        rx_pos: Position3D,
        target_pos: np.ndarray,
        frequencies_hz: np.ndarray,
        p_tx_watts: float = 1000.0
    ) -> Dict[str, Any]:
        num_freqs = len(frequencies_hz)
        num_pos = len(target_pos)
        
        f_pwr_matrix = np.zeros((num_freqs, num_pos))
        snr_lin_matrix = np.zeros((num_freqs, num_pos))
        
        for idx, freq in enumerate(frequencies_hz):
            res = self.model.compute(tx_pos, rx_pos, target_pos, frequency_hz=freq)
            pwr_res = calculate_bistatic_received_power(res, p_tx_watts=p_tx_watts, frequency_hz=freq)
            f_pwr_matrix[idx, :] = res["f_total_pwr"]
            snr_lin_matrix[idx, :] = 10.0**(pwr_res["snr_multipath_db"] / 10.0)
            
        sc_snr = DiversityCombiner.selection_combining(snr_lin_matrix)
        mrc_snr = DiversityCombiner.maximal_ratio_combining(snr_lin_matrix)
        egc_snr = DiversityCombiner.equal_gain_combining(snr_lin_matrix)
        
        sc_snr_db = 10.0 * np.log10(np.maximum(sc_snr, 1e-12))
        mrc_snr_db = 10.0 * np.log10(np.maximum(mrc_snr, 1e-12))
        egc_snr_db = 10.0 * np.log10(np.maximum(egc_snr, 1e-12))
        single_branch_db = 10.0 * np.log10(np.maximum(snr_lin_matrix[0, :], 1e-12))
        
        div_gain_sc_db = sc_snr_db - single_branch_db
        div_gain_mrc_db = mrc_snr_db - single_branch_db
        
        corr_matrix = np.corrcoef(f_pwr_matrix)
        
        return {
            "frequencies_hz": frequencies_hz,
            "f_pwr_matrix": f_pwr_matrix,
            "snr_lin_matrix": snr_lin_matrix,
            "single_branch_db": single_branch_db,
            "sc_snr_db": sc_snr_db,
            "mrc_snr_db": mrc_snr_db,
            "egc_snr_db": egc_snr_db,
            "div_gain_sc_db": div_gain_sc_db,
            "div_gain_mrc_db": div_gain_mrc_db,
            "corr_matrix": corr_matrix,
        }

class SpatialDiversityAnalyzer:
    def __init__(self, two_ray_model: BistaticTwoRayModel):
        self.model = two_ray_model

    def evaluate_rx_positions(
        self,
        tx_pos: Position3D,
        rx_positions: List[Position3D],
        target_pos: np.ndarray,
        p_tx_watts: float = 1000.0
    ) -> Dict[str, Any]:
        num_rx = len(rx_positions)
        num_pos = len(target_pos)
        
        f_pwr_matrix = np.zeros((num_rx, num_pos))
        snr_lin_matrix = np.zeros((num_rx, num_pos))
        
        for idx, rx in enumerate(rx_positions):
            res = self.model.compute(tx_pos, rx, target_pos)
            pwr_res = calculate_bistatic_received_power(res, p_tx_watts=p_tx_watts, frequency_hz=self.model.frequency_hz)
            f_pwr_matrix[idx, :] = res["f_total_pwr"]
            snr_lin_matrix[idx, :] = 10.0**(pwr_res["snr_multipath_db"] / 10.0)
            
        sc_snr = DiversityCombiner.selection_combining(snr_lin_matrix)
        mrc_snr = DiversityCombiner.maximal_ratio_combining(snr_lin_matrix)
        egc_snr = DiversityCombiner.equal_gain_combining(snr_lin_matrix)
        
        sc_snr_db = 10.0 * np.log10(np.maximum(sc_snr, 1e-12))
        mrc_snr_db = 10.0 * np.log10(np.maximum(mrc_snr, 1e-12))
        egc_snr_db = 10.0 * np.log10(np.maximum(egc_snr, 1e-12))
        single_branch_db = 10.0 * np.log10(np.maximum(snr_lin_matrix[0, :], 1e-12))
        
        div_gain_sc_db = sc_snr_db - single_branch_db
        div_gain_mrc_db = mrc_snr_db - single_branch_db
        
        corr_matrix = np.corrcoef(f_pwr_matrix)
        
        return {
            "rx_positions": rx_positions,
            "f_pwr_matrix": f_pwr_matrix,
            "snr_lin_matrix": snr_lin_matrix,
            "single_branch_db": single_branch_db,
            "sc_snr_db": sc_snr_db,
            "mrc_snr_db": mrc_snr_db,
            "egc_snr_db": egc_snr_db,
            "div_gain_sc_db": div_gain_sc_db,
            "div_gain_mrc_db": div_gain_mrc_db,
            "corr_matrix": corr_matrix,
        }
