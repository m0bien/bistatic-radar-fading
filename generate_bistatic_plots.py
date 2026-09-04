import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.geometry import Position3D
from src.reflection import Polarization, GroundMedium
from src.simulator import BistaticRadarSimulator
from src.radar_equation import calculate_bistatic_received_power

def main():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fc_hz = 98.0e6
    tx_pos = Position3D(x=-20000.0, y=0.0, z=120.0)
    rx_pos = Position3D(x=20000.0, y=0.0, z=15.0)
    
    sim = BistaticRadarSimulator(
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        frequency_hz=fc_hz,
        polarization=Polarization.VERTICAL,
        ground_type=GroundMedium.MEDIUM_GROUND,
        rms_roughness_m=0.05
    )
    
    # 1. Generate 2D Grid of Bistatic Angle and Propagation Factor F^4
    grid_res = sim.run_2d_spatial_grid(
        x_range=(-60000.0, 60000.0),
        y_range=(-60000.0, 60000.0),
        target_height_m=2500.0,
        grid_resolution_m=600.0
    )
    
    f_db_2d = grid_res["f_total_db_2d"]
    beta_2d = grid_res["bistatic_angle_2d"]
    xx = grid_res["xx"] / 1e3
    yy = grid_res["yy"] / 1e3
    
    # Plot 7: Fading & Bistatic Angle Mapping
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: 2D Spatial Multipath Fading with Bistatic Angle Contours
    ax1 = axes[0]
    im1 = ax1.pcolormesh(xx, yy, f_db_2d, cmap="coolwarm", shading="auto", vmin=-25, vmax=12)
    cb1 = fig.colorbar(im1, ax=ax1)
    cb1.set_label(r"Bistatic Pattern Propagation Factor $F^4$ (dB)", fontsize=11)
    
    # Overlay isorange / bistatic angle contours
    cs = ax1.contour(xx, yy, beta_2d, levels=[20, 40, 60, 90, 120, 150], colors="black", linewidths=1.0, linestyles="--")
    ax1.clabel(cs, inline=True, fmt=r"$\beta=%d^\circ$", fontsize=9)
    
    ax1.plot(tx_pos.x/1e3, tx_pos.y/1e3, "y^", markersize=12, markeredgecolor="k", label=f"Tx (FM Tower, h={tx_pos.z:.0f}m)")
    ax1.plot(rx_pos.x/1e3, rx_pos.y/1e3, "gv", markersize=12, markeredgecolor="k", label=f"Rx (Radar Mast, h={rx_pos.z:.0f}m)")
    ax1.plot([tx_pos.x/1e3, rx_pos.x/1e3], [tx_pos.y/1e3, rx_pos.y/1e3], "k:", lw=1.5, label="Baseline (40 km)")
    ax1.set_title(r"2D Fading Distribution with Bistatic Angle $\beta$ Contours", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X Coordinate (km)", fontsize=11)
    ax1.set_ylabel("Y Coordinate (km)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.axis("equal")
    ax1.grid(True, alpha=0.2)
    
    # Right: Fading Factor vs Bistatic Angle (scatter / sorted profile along circular trajectory)
    ax2 = axes[1]
    ang_res = sim.run_angular_sweep(radius_m=45000.0, target_height_m=2500.0, num_angles=1500)
    beta_ang = ang_res["two_ray_res"]["geometry"]["bistatic_angle_deg"]
    f_ang_db = ang_res["two_ray_res"]["f_total_db"]
    snr_ang_db = ang_res["pwr_res"]["snr_multipath_db"]
    
    # Sort by bistatic angle for sorted visualization
    sort_idx = np.argsort(beta_ang)
    ax2.plot(beta_ang[sort_idx], f_ang_db[sort_idx], "b.", markersize=2.5, alpha=0.5, label=r"Propagation Factor $F^4$ (dB)")
    ax2.axhline(0, color="k", linestyle="--", alpha=0.6, label="Free Space (0 dB)")
    ax2.set_title(r"Multipath Fading Depth vs Bistatic Angle $\beta$", fontsize=12, fontweight="bold")
    ax2.set_xlabel(r"Bistatic Angle $\beta$ (degrees)", fontsize=11)
    ax2.set_ylabel(r"Pattern Propagation Factor $F^4$ (dB)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(np.min(beta_ang) - 2, np.max(beta_ang) + 2)
    ax2.legend(loc="lower right", fontsize=9)
    
    plt.tight_layout()
    plot7_path = results_dir / "7_bistatic_angle_fading_analysis.png"
    plt.savefig(plot7_path, dpi=300)
    plt.close()
    print(f"Saved: {plot7_path}")
    
    # Copy to artifact folder
    artifact_dir = Path(r"C:\Users\m_mub\.gemini\antigravity\brain\2ad4a709-e895-4937-b6cd-fb0ad49bd9a2")
    for p in results_dir.glob("*.png"):
        dest = artifact_dir / p.name
        with open(p, "rb") as f_src, open(dest, "wb") as f_dst:
            f_dst.write(f_src.read())

if __name__ == "__main__":
    main()
