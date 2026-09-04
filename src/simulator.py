import numpy as np
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
