"""Bistatic Radar Two-Ray Fading & Diversity Analysis Package."""
from .geometry import Position3D, calculate_bistatic_geometry
from .reflection import GroundMedium, Polarization, fresnel_reflection_coefficient, surface_roughness_factor
from .propagation import compute_two_ray_pattern_factors, BistaticTwoRayModel
from .radar_equation import bistatic_rcs_model, calculate_bistatic_received_power
from .diversity import FrequencyDiversityAnalyzer, SpatialDiversityAnalyzer, DiversityCombiner
from .simulator import BistaticRadarSimulator
