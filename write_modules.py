import os
from pathlib import Path

base_dir = Path(r"C:\Users\m_mub\.gemini\antigravity\scratch\bistatic_radar_fading\src")
base_dir.mkdir(parents=True, exist_ok=True)

files = {}

files["__init__.py"] = """\"\"\"Bistatic Radar Two-Ray Fading & Diversity Analysis Package.\"\"\"
from .geometry import Position3D, calculate_bistatic_geometry
from .reflection import GroundMedium, Polarization, fresnel_reflection_coefficient, surface_roughness_factor
from .propagation import compute_two_ray_pattern_factors, BistaticTwoRayModel
from .radar_equation import bistatic_rcs_model, calculate_bistatic_received_power
from .diversity import FrequencyDiversityAnalyzer, SpatialDiversityAnalyzer, DiversityCombiner
from .simulator import BistaticRadarSimulator
"""

files["geometry.py"] = """import numpy as np
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
    \"\"\"
    Computes direct and ground-reflected path lengths, grazing angles, 
    specular reflection points, and bistatic angles for 3D bistatic radar.
    Ground plane is at z = 0.
    \"\"\"
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
"""

files["reflection.py"] = """import numpy as np
from enum import Enum
from typing import Union

class Polarization(Enum):
    HORIZONTAL = "horizontal"  # TE
    VERTICAL = "vertical"      # TM
    CIRCULAR = "circular"

class GroundMedium(Enum):
    SEAWATER = "seawater"          # eps_r = 80, sigma = 4.0 S/m
    FRESH_WATER = "fresh_water"    # eps_r = 81, sigma = 0.01 S/m
    WET_GROUND = "wet_ground"      # eps_r = 25, sigma = 0.02 S/m
    MEDIUM_GROUND = "medium_ground"# eps_r = 15, sigma = 0.005 S/m
    DRY_GROUND = "dry_ground"      # eps_r = 4,  sigma = 0.001 S/m
    PERFECT_CONDUCTOR = "pec"      # Gamma_H = -1, Gamma_V = +1

GROUND_PROPERTIES = {
    GroundMedium.SEAWATER: (80.0, 4.0),
    GroundMedium.FRESH_WATER: (81.0, 0.01),
    GroundMedium.WET_GROUND: (25.0, 0.02),
    GroundMedium.MEDIUM_GROUND: (15.0, 0.005),
    GroundMedium.DRY_GROUND: (4.0, 0.001),
    GroundMedium.PERFECT_CONDUCTOR: (np.inf, np.inf),
}

def complex_permittivity(
    ground_type: GroundMedium,
    frequency_hz: float
) -> complex:
    \"\"\"Calculates complex relative permittivity eps_c = eps_r - j * sigma / (2*pi*f*eps_0).\"\"\"
    if ground_type == GroundMedium.PERFECT_CONDUCTOR:
        return complex(1e12, -1e12)
    eps_r, sigma = GROUND_PROPERTIES[ground_type]
    eps_0 = 8.8541878128e-12
    omega = 2.0 * np.pi * frequency_hz
    eps_c = eps_r - 1j * (sigma / (omega * eps_0 + 1e-18))
    return eps_c

def fresnel_reflection_coefficient(
    psi_rad: Union[float, np.ndarray],
    frequency_hz: float,
    polarization: Polarization = Polarization.HORIZONTAL,
    ground_type: GroundMedium = GroundMedium.MEDIUM_GROUND
) -> Union[complex, np.ndarray]:
    \"\"\"
    Computes Fresnel reflection coefficient Gamma for a given grazing angle psi (radians).
    \"\"\"
    if ground_type == GroundMedium.PERFECT_CONDUCTOR:
        if polarization == Polarization.HORIZONTAL:
            return -1.0 + 0j
        else:
            return 1.0 + 0j

    eps_c = complex_permittivity(ground_type, frequency_hz)
    sin_psi = np.sin(psi_rad)
    cos_psi = np.cos(psi_rad)
    
    radicand = eps_c - cos_psi**2
    sqrt_radicand = np.sqrt(radicand)
    
    if polarization == Polarization.HORIZONTAL:
        gamma = (sin_psi - sqrt_radicand) / (sin_psi + sqrt_radicand + 1e-12)
    elif polarization == Polarization.VERTICAL:
        gamma = (eps_c * sin_psi - sqrt_radicand) / (eps_c * sin_psi + sqrt_radicand + 1e-12)
    elif polarization == Polarization.CIRCULAR:
        gamma_h = (sin_psi - sqrt_radicand) / (sin_psi + sqrt_radicand + 1e-12)
        gamma_v = (eps_c * sin_psi - sqrt_radicand) / (eps_c * sin_psi + sqrt_radicand + 1e-12)
        gamma = 0.5 * (gamma_h + gamma_v)
    else:
        raise ValueError(f"Unknown polarization {polarization}")
        
    return gamma

def surface_roughness_factor(
    psi_rad: Union[float, np.ndarray],
    frequency_hz: float,
    rms_roughness_m: float = 0.0
) -> Union[float, np.ndarray]:
    \"\"\"
    Miller-Brown roughness reduction factor rho_s = exp(-0.5 * (4*pi*sigma_h * sin(psi) / lambda)^2)
    \"\"\"
    if rms_roughness_m <= 0.0:
        return np.ones_like(psi_rad, dtype=np.float64) if isinstance(psi_rad, np.ndarray) else 1.0
    
    c = 299792458.0
    wavelength = c / frequency_hz
    k = 2.0 * np.pi / wavelength
    param = 2.0 * k * rms_roughness_m * np.sin(psi_rad)
    rho_s = np.exp(-0.5 * (param**2))
    return rho_s
"""

files["propagation.py"] = """import numpy as np
from typing import Dict, Any, Union
from .geometry import Position3D, calculate_bistatic_geometry
from .reflection import (
    Polarization,
    GroundMedium,
    fresnel_reflection_coefficient,
    surface_roughness_factor
)

class BistaticTwoRayModel:
    \"\"\"
    Computes the Bistatic Radar Pattern Propagation Factors:
    F_t = 1 + Gamma_t * rho_st * exp(-j * k * delta_r_tx)
    F_r = 1 + Gamma_r * rho_sr * exp(-j * k * delta_r_rx)
    Total Bistatic Propagation Factor F = F_t * F_r
    Total Two-Way Propagation Power Factor F^4 = |F_t|^2 * |F_r|^2
    \"\"\"
    def __init__(
        self,
        frequency_hz: float = 3.0e9,
        polarization: Polarization = Polarization.HORIZONTAL,
        ground_type: GroundMedium = GroundMedium.MEDIUM_GROUND,
        rms_roughness_m: float = 0.0,
    ):
        self.frequency_hz = frequency_hz
        self.polarization = polarization
        self.ground_type = ground_type
        self.rms_roughness_m = rms_roughness_m
        self.c = 299792458.0

    @property
    def wavelength(self) -> float:
        return self.c / self.frequency_hz

    @property
    def wavenumber(self) -> float:
        return 2.0 * np.pi / self.wavelength

    def compute(
        self,
        tx_pos: Position3D,
        rx_pos: Position3D,
        target_pos: Union[Position3D, np.ndarray],
        frequency_hz: float = None
    ) -> Dict[str, Any]:
        f_hz = self.frequency_hz if frequency_hz is None else frequency_hz
        k = 2.0 * np.pi * f_hz / self.c
        
        geo = calculate_bistatic_geometry(tx_pos, rx_pos, target_pos)
        
        # Reflection coefficients
        gamma_t = fresnel_reflection_coefficient(
            geo["psi_tx_rad"], f_hz, self.polarization, self.ground_type
        )
        gamma_r = fresnel_reflection_coefficient(
            geo["psi_rx_rad"], f_hz, self.polarization, self.ground_type
        )
        
        # Roughness factors
        rho_t = surface_roughness_factor(geo["psi_tx_rad"], f_hz, self.rms_roughness_m)
        rho_r = surface_roughness_factor(geo["psi_rx_rad"], f_hz, self.rms_roughness_m)
        
        # Phase shifts
        phase_tx = k * geo["delta_r_tx"]
        phase_rx = k * geo["delta_r_rx"]
        
        # Complex propagation factors
        f_tx = 1.0 + gamma_t * rho_t * np.exp(-1j * phase_tx)
        f_rx = 1.0 + gamma_r * rho_r * np.exp(-1j * phase_rx)
        
        # Total propagation factor
        f_total = f_tx * f_rx
        
        # Power factors
        f_tx_sq = np.abs(f_tx)**2
        f_rx_sq = np.abs(f_rx)**2
        f_total_pwr = f_tx_sq * f_rx_sq
        f_total_db = 10.0 * np.log10(np.maximum(f_total_pwr, 1e-15))
        fading_depth_db = -f_total_db
        
        return {
            "geometry": geo,
            "gamma_t": gamma_t,
            "gamma_r": gamma_r,
            "f_tx": f_tx,
            "f_rx": f_rx,
            "f_total": f_total,
            "f_tx_sq": f_tx_sq,
            "f_rx_sq": f_rx_sq,
            "f_total_pwr": f_total_pwr,
            "f_total_db": f_total_db,
            "fading_depth_db": fading_depth_db,
        }

def compute_two_ray_pattern_factors(
    tx_pos: Position3D,
    rx_pos: Position3D,
    target_pos: Union[Position3D, np.ndarray],
    frequency_hz: float = 3e9,
    polarization: Polarization = Polarization.HORIZONTAL,
    ground_type: GroundMedium = GroundMedium.MEDIUM_GROUND,
    rms_roughness_m: float = 0.0
) -> Dict[str, Any]:
    model = BistaticTwoRayModel(
        frequency_hz=frequency_hz,
        polarization=polarization,
        ground_type=ground_type,
        rms_roughness_m=rms_roughness_m
    )
    return model.compute(tx_pos, rx_pos, target_pos)
"""

files["radar_equation.py"] = """import numpy as np
from typing import Dict, Any, Union
from .geometry import Position3D
from .propagation import BistaticTwoRayModel

def bistatic_rcs_model(
    bistatic_angle_deg: Union[float, np.ndarray],
    sigma_monostatic: float = 1.0,
    target_type: str = "sphere"
) -> Union[float, np.ndarray]:
    \"\"\"
    Bistatic Radar Cross Section (RCS) model.
    \"\"\"
    beta_rad = np.radians(bistatic_angle_deg)
    if target_type == "sphere" or target_type == "isotropic":
        forward_lobe = np.exp(-((180.0 - bistatic_angle_deg)/10.0)**2) * 10.0
        return np.maximum(sigma_monostatic * (np.cos(beta_rad / 2.0)**2) + forward_lobe, 0.01)
    elif target_type == "constant":
        return np.ones_like(bistatic_angle_deg) * sigma_monostatic if isinstance(bistatic_angle_deg, np.ndarray) else sigma_monostatic
    else:
        return np.ones_like(bistatic_angle_deg) * sigma_monostatic

def calculate_bistatic_received_power(
    two_ray_result: Dict[str, Any],
    p_tx_watts: float = 1000.0,
    g_tx_db: float = 30.0,
    g_rx_db: float = 30.0,
    sigma_m2: float = 1.0,
    target_rcs_type: str = "sphere",
    noise_figure_db: float = 4.0,
    bandwidth_hz: float = 10e6,
    frequency_hz: float = 3e9
) -> Dict[str, Any]:
    c = 299792458.0
    wavelength = c / frequency_hz
    
    g_tx_lin = 10.0**(g_tx_db / 10.0)
    g_rx_lin = 10.0**(g_rx_db / 10.0)
    
    geo = two_ray_result["geometry"]
    r_tx = geo["r_tx_direct"]
    r_rx = geo["r_rx_direct"]
    beta_deg = geo["bistatic_angle_deg"]
    
    sigma_b = bistatic_rcs_model(beta_deg, sigma_monostatic=sigma_m2, target_type=target_rcs_type)
    f_total_pwr = two_ray_result["f_total_pwr"]
    
    # Free space received power
    numerator = p_tx_watts * g_tx_lin * g_rx_lin * (wavelength**2) * sigma_b
    denominator = ((4.0 * np.pi)**3) * (r_tx**2) * (r_rx**2) + 1e-18
    p_rx_freespace = numerator / denominator
    
    # Multipath received power
    p_rx_multipath = p_rx_freespace * f_total_pwr
    
    # Thermal Noise
    k_boltzmann = 1.380649e-23
    t_0 = 290.0
    f_noise_lin = 10.0**(noise_figure_db / 10.0)
    p_noise = k_boltzmann * t_0 * bandwidth_hz * f_noise_lin
    
    snr_freespace_lin = p_rx_freespace / p_noise
    snr_multipath_lin = p_rx_multipath / p_noise
    
    snr_freespace_db = 10.0 * np.log10(np.maximum(snr_freespace_lin, 1e-12))
    snr_multipath_db = 10.0 * np.log10(np.maximum(snr_multipath_lin, 1e-12))
    
    return {
        "p_rx_freespace_watts": p_rx_freespace,
        "p_rx_multipath_watts": p_rx_multipath,
        "p_rx_multipath_dbm": 10.0 * np.log10(np.maximum(p_rx_multipath, 1e-20) * 1000.0),
        "snr_freespace_db": snr_freespace_db,
        "snr_multipath_db": snr_multipath_db,
        "p_noise_watts": p_noise,
        "sigma_b": sigma_b,
    }
"""

files["diversity.py"] = """import numpy as np
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
"""

files["simulator.py"] = """import numpy as np
from typing import Dict, Any, List
from .geometry import Position3D
from .reflection import Polarization, GroundMedium
from .propagation import BistaticTwoRayModel
from .radar_equation import calculate_bistatic_received_power
from .diversity import FrequencyDiversityAnalyzer, SpatialDiversityAnalyzer

class BistaticRadarSimulator:
    def __init__(
        self,
        tx_pos: Position3D = Position3D(x=-10000.0, y=0.0, z=25.0),
        rx_pos: Position3D = Position3D(x=10000.0, y=0.0, z=15.0),
        frequency_hz: float = 3.0e9,
        polarization: Polarization = Polarization.HORIZONTAL,
        ground_type: GroundMedium = GroundMedium.MEDIUM_GROUND,
        rms_roughness_m: float = 0.02
    ):
        self.tx_pos = tx_pos
        self.rx_pos = rx_pos
        self.frequency_hz = frequency_hz
        self.polarization = polarization
        self.ground_type = ground_type
        self.rms_roughness_m = rms_roughness_m
        
        self.model = BistaticTwoRayModel(
            frequency_hz=frequency_hz,
            polarization=polarization,
            ground_type=ground_type,
            rms_roughness_m=rms_roughness_m
        )

    def run_angular_sweep(
        self,
        radius_m: float = 30000.0,
        target_height_m: float = 1000.0,
        num_angles: int = 720,
        center_x: float = 0.0,
        center_y: float = 0.0
    ) -> Dict[str, Any]:
        azimuth_deg = np.linspace(0.0, 360.0, num_angles, endpoint=False)
        azimuth_rad = np.radians(azimuth_deg)
        
        x_tgt = center_x + radius_m * np.cos(azimuth_rad)
        y_tgt = center_y + radius_m * np.sin(azimuth_rad)
        z_tgt = np.full_like(x_tgt, target_height_m)
        
        target_grid = np.column_stack((x_tgt, y_tgt, z_tgt))
        
        two_ray_res = self.model.compute(self.tx_pos, self.rx_pos, target_grid)
        pwr_res = calculate_bistatic_received_power(two_ray_res, frequency_hz=self.frequency_hz)
        
        return {
            "azimuth_deg": azimuth_deg,
            "target_grid": target_grid,
            "radius_m": radius_m,
            "target_height_m": target_height_m,
            "two_ray_res": two_ray_res,
            "pwr_res": pwr_res
        }

    def run_2d_spatial_grid(
        self,
        x_range: tuple = (-40000.0, 40000.0),
        y_range: tuple = (-40000.0, 40000.0),
        target_height_m: float = 1000.0,
        grid_resolution_m: float = 400.0
    ) -> Dict[str, Any]:
        x = np.arange(x_range[0], x_range[1] + grid_resolution_m, grid_resolution_m)
        y = np.arange(y_range[0], y_range[1] + grid_resolution_m, grid_resolution_m)
        xx, yy = np.meshgrid(x, y)
        
        zz = np.full_like(xx, target_height_m)
        target_grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        
        two_ray_res = self.model.compute(self.tx_pos, self.rx_pos, target_grid)
        pwr_res = calculate_bistatic_received_power(two_ray_res, frequency_hz=self.frequency_hz)
        
        f_total_db_2d = two_ray_res["f_total_db"].reshape(xx.shape)
        snr_multipath_db_2d = pwr_res["snr_multipath_db"].reshape(xx.shape)
        bistatic_angle_2d = two_ray_res["geometry"]["bistatic_angle_deg"].reshape(xx.shape)
        
        return {
            "x": x,
            "y": y,
            "xx": xx,
            "yy": yy,
            "f_total_db_2d": f_total_db_2d,
            "snr_multipath_db_2d": snr_multipath_db_2d,
            "bistatic_angle_2d": bistatic_angle_2d
        }

    def run_diversity_angular_analysis(
        self,
        radius_m: float = 30000.0,
        target_height_m: float = 1000.0,
        num_angles: int = 720,
        freq_offsets_mhz: List[float] = [0.0, 20.0, 50.0, 100.0],
        rx_height_offsets_m: List[float] = [0.0, 4.0, 8.0, 12.0]
    ) -> Dict[str, Any]:
        azimuth_deg = np.linspace(0.0, 360.0, num_angles, endpoint=False)
        azimuth_rad = np.radians(azimuth_deg)
        x_tgt = radius_m * np.cos(azimuth_rad)
        y_tgt = radius_m * np.sin(azimuth_rad)
        z_tgt = np.full_like(x_tgt, target_height_m)
        target_grid = np.column_stack((x_tgt, y_tgt, z_tgt))
        
        # Frequency Diversity
        freq_analyzer = FrequencyDiversityAnalyzer(self.model)
        freqs_hz = np.array([self.frequency_hz + df * 1e6 for df in freq_offsets_mhz])
        freq_res = freq_analyzer.evaluate_frequency_grid(
            self.tx_pos, self.rx_pos, target_grid, freqs_hz
        )
        
        # Spatial Diversity
        spatial_analyzer = SpatialDiversityAnalyzer(self.model)
        rx_positions = [
            Position3D(self.rx_pos.x, self.rx_pos.y, self.rx_pos.z + dh)
            for dh in rx_height_offsets_m
        ]
        spatial_res = spatial_analyzer.evaluate_rx_positions(
            self.tx_pos, rx_positions, target_grid
        )
        
        return {
            "azimuth_deg": azimuth_deg,
            "freq_offsets_mhz": freq_offsets_mhz,
            "rx_height_offsets_m": rx_height_offsets_m,
            "freq_diversity": freq_res,
            "spatial_diversity": spatial_res
        }
"""

for fname, content in files.items():
    fpath = base_dir / fname
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Written: {fname}")
