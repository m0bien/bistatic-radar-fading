import numpy as np
from typing import Dict, Any, Union
from .geometry import Position3D
from .propagation import BistaticTwoRayModel

def bistatic_rcs_model(
    bistatic_angle_deg: Union[float, np.ndarray],
    sigma_monostatic: float = 1.0,
    target_type: str = "sphere"
) -> Union[float, np.ndarray]:
    """
    Bistatic Radar Cross Section (RCS) model.
    """
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
