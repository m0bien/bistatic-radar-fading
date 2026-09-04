import numpy as np
from typing import Dict, Any, Union
from .geometry import Position3D, calculate_bistatic_geometry
from .reflection import (
    Polarization,
    GroundMedium,
    fresnel_reflection_coefficient,
    surface_roughness_factor
)

class BistaticTwoRayModel:
    """
    Computes the Bistatic Radar Pattern Propagation Factors:
    F_t = 1 + Gamma_t * rho_st * exp(-j * k * delta_r_tx)
    F_r = 1 + Gamma_r * rho_sr * exp(-j * k * delta_r_rx)
    Total Bistatic Propagation Factor F = F_t * F_r
    Total Two-Way Propagation Power Factor F^4 = |F_t|^2 * |F_r|^2
    """
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
