import numpy as np
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
    """Calculates complex relative permittivity eps_c = eps_r - j * sigma / (2*pi*f*eps_0)."""
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
    """
    Computes Fresnel reflection coefficient Gamma for a given grazing angle psi (radians).
    """
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
    """
    Miller-Brown roughness reduction factor rho_s = exp(-0.5 * (4*pi*sigma_h * sin(psi) / lambda)^2)
    """
    if rms_roughness_m <= 0.0:
        return np.ones_like(psi_rad, dtype=np.float64) if isinstance(psi_rad, np.ndarray) else 1.0
    
    c = 299792458.0
    wavelength = c / frequency_hz
    k = 2.0 * np.pi / wavelength
    param = 2.0 * k * rms_roughness_m * np.sin(psi_rad)
    rho_s = np.exp(-0.5 * (param**2))
    return rho_s
