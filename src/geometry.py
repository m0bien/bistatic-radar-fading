import numpy as np
from dataclasses import dataclass
from typing import Union, Dict, Any

@dataclass
class Position3D:
    x: float
    y: float
    z: float  # Height above ground plane (z >= 0)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

def calculate_bistatic_geometry(
    tx_pos: Position3D,
    rx_pos: Position3D,
    target_pos: Union[Position3D, np.ndarray]
) -> Dict[str, Any]:
    """
    Computes direct and ground-reflected path lengths, grazing angles, 
    specular reflection points, and bistatic angles for 3D bistatic radar.
    Ground plane is at z = 0.
    """
    tx = tx_pos.to_array()
    rx = rx_pos.to_array()
    
    if isinstance(target_pos, Position3D):
        tgt = target_pos.to_array()
        is_array = False
    else:
        tgt = np.asarray(target_pos, dtype=np.float64)
        is_array = (tgt.ndim > 1)

    # Direct Paths
    vec_tx_tgt = tgt - tx
    r_tx_direct = np.linalg.norm(vec_tx_tgt, axis=-1)
    
    vec_tgt_rx = rx - tgt
    r_rx_direct = np.linalg.norm(vec_tgt_rx, axis=-1)
    
    vec_tx_rx = rx - tx
    baseline_length = float(np.linalg.norm(vec_tx_rx))
    
    # Bistatic Angle beta (angle subtended at target by Tx and Rx)
    vec_tgt_tx = -vec_tx_tgt
    dot_prod = np.sum(vec_tgt_tx * vec_tgt_rx, axis=-1)
    cos_beta = dot_prod / (r_tx_direct * r_rx_direct + 1e-12)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    bistatic_angle_rad = np.arccos(cos_beta)
    bistatic_angle_deg = np.degrees(bistatic_angle_rad)
    
    # Ground Reflected Paths via Image Theory
    tx_img = np.array([tx[0], tx[1], -tx[2]])
    vec_tx_img_tgt = tgt - tx_img
    r_tx_refl = np.linalg.norm(vec_tx_img_tgt, axis=-1)
    
    rx_img = np.array([rx[0], rx[1], -rx[2]])
    vec_tgt_rx_img = rx_img - tgt
    r_rx_refl = np.linalg.norm(vec_tgt_rx_img, axis=-1)
    
    delta_r_tx = r_tx_refl - r_tx_direct
    delta_r_rx = r_rx_refl - r_rx_direct
    
    # Grazing Angles
    d_horiz_tx = np.linalg.norm(tgt[..., :2] - tx[:2], axis=-1)
    psi_tx_rad = np.arctan2(tx[2] + tgt[..., 2], d_horiz_tx + 1e-12)
    
    d_horiz_rx = np.linalg.norm(rx[:2] - tgt[..., :2], axis=-1)
    psi_rx_rad = np.arctan2(rx[2] + tgt[..., 2], d_horiz_rx + 1e-12)
    
    # Specular Reflection Points (on z=0)
    alpha_tx = tx[2] / (tx[2] + tgt[..., 2] + 1e-12)
    refl_pt_tx = tx[:2] + (tgt[..., :2] - tx[:2]) * (alpha_tx[..., np.newaxis] if is_array else alpha_tx)
    
    alpha_rx = rx[2] / (rx[2] + tgt[..., 2] + 1e-12)
    refl_pt_rx = rx[:2] + (tgt[..., :2] - rx[:2]) * (alpha_rx[..., np.newaxis] if is_array else alpha_rx)
    
    return {
        "r_tx_direct": r_tx_direct,
        "r_rx_direct": r_rx_direct,
        "r_tx_refl": r_tx_refl,
        "r_rx_refl": r_rx_refl,
        "delta_r_tx": delta_r_tx,
        "delta_r_rx": delta_r_rx,
        "psi_tx_rad": psi_tx_rad,
        "psi_rx_rad": psi_rx_rad,
        "bistatic_angle_deg": bistatic_angle_deg,
        "bistatic_angle_rad": bistatic_angle_rad,
        "baseline_length": baseline_length,
        "refl_pt_tx": refl_pt_tx,
        "refl_pt_rx": refl_pt_rx,
    }
