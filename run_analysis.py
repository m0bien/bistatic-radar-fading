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
from src.diversity import DiversityCombiner, FrequencyDiversityAnalyzer, SpatialDiversityAnalyzer
from src.radar_equation import calculate_bistatic_received_power

def main():
    print("=" * 75)
    print("  VHF FM-BAND BISTATIC RADAR (98 MHz, 88-108 MHz DIVERSITY) ANALYSIS")
    print("=" * 75)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup VHF FM Broadcast Bistatic Radar Parameters
    fc_hz = 98.0e6  # 98 MHz nominal carrier
    tx_pos = Position3D(x=-20000.0, y=0.0, z=120.0)   # FM Broadcast Tower: 120m mast, 20km West
    rx_pos = Position3D(x=20000.0, y=0.0, z=15.0)     # Surveillance Rx: 15m mast, 20km East
    ground_type = GroundMedium.MEDIUM_GROUND          # eps_r=15, sigma=0.005 S/m
    polarization = Polarization.VERTICAL               # FM broadcast commonly mixed / vertical / horizontal
    rms_roughness_m = 0.05                             # 5 cm ground roughness (minor effect at lambda = 3.06m)
    
    # Available FM Band Channels (88 - 108 MHz)
    fm_frequencies_hz = np.array([88.5e6, 93.0e6, 98.0e6, 103.0e6, 107.5e6])
    
    sim = BistaticRadarSimulator(
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        frequency_hz=fc_hz,
        polarization=polarization,
        ground_type=ground_type,
        rms_roughness_m=rms_roughness_m
    )
    
    c = 299792458.0
    wavelength = c / fc_hz
    baseline_km = np.linalg.norm(rx_pos.to_array() - tx_pos.to_array()) / 1e3
    
    print(f"Bistatic System: VHF FM-band Passive / Bistatic Radar")
    print(f"Nominal Carrier:    {fc_hz/1e6:.1f} MHz (Wavelength lambda = {wavelength:.2f} m)")
    print(f"FM Diversity Band:  88.0 MHz to 108.0 MHz (Total Bandwidth: 20 MHz, Fractional: 20.4%)")
    print(f"FM Station Channels:{[f'{f/1e6:.1f} MHz' for f in fm_frequencies_hz]}")
    print(f"Tx (Broadcast Mast):({tx_pos.x/1e3:.1f} km, {tx_pos.y/1e3:.1f} km, Height = {tx_pos.z:.1f} m)")
    print(f"Rx (Radar Receiver):({rx_pos.x/1e3:.1f} km, {rx_pos.y/1e3:.1f} km, Height = {rx_pos.z:.1f} m)")
    print(f"Baseline Distance:  {baseline_km:.1f} km")
    print(f"Polarization:       {polarization.value}, Ground: {ground_type.value}")
    print("-" * 75)

    # =========================================================================
    # PART 1: 360-DEGREE ANGULAR FADING AT 98 MHz
    # =========================================================================
    print("Running 360-degree Angular Fading Sweep at 98 MHz...")
    radius_m = 45000.0        # Target distance 45 km from baseline midpoint
    target_height_m = 2500.0  # Commercial / airborne target altitude 2.5 km
    num_angles = 1200
    
    ang_res = sim.run_angular_sweep(
        radius_m=radius_m,
        target_height_m=target_height_m,
        num_angles=num_angles
    )
    
    azimuth = ang_res["azimuth_deg"]
    f_total_db = ang_res["two_ray_res"]["f_total_db"]
    snr_multipath_db = ang_res["pwr_res"]["snr_multipath_db"]
    snr_freespace_db = ang_res["pwr_res"]["snr_freespace_db"]
    bistatic_angle = ang_res["two_ray_res"]["geometry"]["bistatic_angle_deg"]
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(azimuth, f_total_db, "b-", lw=1.2, label=r"Bistatic Multipath Factor $F^4$ (dB)")
    plt.axhline(0, color="k", linestyle="--", alpha=0.6, label="Free Space (0 dB)")
    plt.title(f"Bistatic Propagation Factor at 98 MHz (\\lambda = {wavelength:.2f} m)\n(R = {radius_m/1e3:.0f} km, Target Altitude = {target_height_m:.0f} m)", fontsize=11, fontweight="bold")
    plt.xlabel("Target Azimuth Angle around Baseline (deg)")
    plt.ylabel(r"Propagation Factor $F^4$ (dB)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    plt.legend(loc="lower right")
    
    plt.subplot(2, 2, 2)
    plt.plot(azimuth, snr_multipath_db, "r-", lw=1.2, label="Two-Ray Multipath SNR (98 MHz)")
    plt.plot(azimuth, snr_freespace_db, "k--", lw=1.5, label="Free Space SNR")
    plt.title("Received SNR vs Azimuth Angle (98 MHz)", fontsize=11, fontweight="bold")
    plt.xlabel("Target Azimuth Angle (deg)")
    plt.ylabel("Received SNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    plt.legend(loc="lower right")
    
    plt.subplot(2, 2, 3)
    plt.plot(azimuth, bistatic_angle, "g-", lw=1.5)
    plt.title(r"Bistatic Angle $\beta$ vs Target Azimuth", fontsize=11, fontweight="bold")
    plt.xlabel("Target Azimuth Angle (deg)")
    plt.ylabel(r"Bistatic Angle $\beta$ (deg)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    
    # Polar Plot
    ax_polar = plt.subplot(2, 2, 4, projection="polar")
    theta_rad = np.radians(azimuth)
    snr_clamped = np.maximum(snr_multipath_db, np.min(snr_freespace_db) - 30.0)
    ax_polar.plot(theta_rad, snr_clamped, "r-", lw=1.0, label="Multipath SNR")
    ax_polar.plot(theta_rad, snr_freespace_db, "k--", lw=1.2, label="Free Space")
    ax_polar.set_title("Polar SNR Coverage Pattern (98 MHz)", fontsize=11, fontweight="bold", va="bottom")
    ax_polar.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot1_path = results_dir / "1_angular_fading_analysis.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")

    # =========================================================================
    # PART 2: 2D SPATIAL HEATMAP AT 98 MHz
    # =========================================================================
    print("Running 2D Spatial Heatmap Simulation (98 MHz)...")
    grid_res = sim.run_2d_spatial_grid(
        x_range=(-60000.0, 60000.0),
        y_range=(-60000.0, 60000.0),
        target_height_m=target_height_m,
        grid_resolution_m=800.0
    )
    
    plt.figure(figsize=(13, 10))
    im = plt.pcolormesh(
        grid_res["x"]/1e3, grid_res["y"]/1e3,
        grid_res["f_total_db_2d"],
        cmap="coolwarm", shading="auto", vmin=-25, vmax=12
    )
    cbar = plt.colorbar(im)
    cbar.set_label(r"Bistatic Pattern Propagation Factor $F^4$ (dB) at 98 MHz", fontsize=12)
    
    plt.plot(tx_pos.x/1e3, tx_pos.y/1e3, "y^", markersize=14, markeredgecolor="black", label=f"Tx FM Tower ({tx_pos.x/1e3:.0f} km, h={tx_pos.z:.0f}m)")
    plt.plot(rx_pos.x/1e3, rx_pos.y/1e3, "gv", markersize=14, markeredgecolor="black", label=f"Rx Station ({rx_pos.x/1e3:.0f} km, h={rx_pos.z:.0f}m)")
    plt.plot([tx_pos.x/1e3, rx_pos.x/1e3], [tx_pos.y/1e3, rx_pos.y/1e3], "k--", lw=1.5, alpha=0.7, label=f"Baseline ({baseline_km:.0f} km)")
    
    circ_angles = np.linspace(0, 2*np.pi, 200)
    plt.plot(radius_m/1e3 * np.cos(circ_angles), radius_m/1e3 * np.sin(circ_angles), "k-.", lw=1.5, label=f"Target Orbit (R={radius_m/1e3:.0f} km)")
    
    plt.title(f"2D Bistatic Radar Multipath Fading ($F^4$ in dB) at 98 MHz (\\lambda = 3.06 m)\nTarget Altitude = {target_height_m:.0f} m, FM Band VHF", fontsize=13, fontweight="bold")
    plt.xlabel("X Coordinate (km)", fontsize=11)
    plt.ylabel("Y Coordinate (km)", fontsize=11)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.axis("equal")
    plt.grid(True, alpha=0.2, linestyle=":")
    
    plot2_path = results_dir / "2_spatial_fading_heatmap.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

    # =========================================================================
    # PART 3: FM-BAND FREQUENCY DIVERSITY (88 - 108 MHz)
    # =========================================================================
    print("Running FM-Band Frequency Diversity Analysis (88 - 108 MHz)...")
    target_grid = ang_res["target_grid"]
    
    freq_analyzer = FrequencyDiversityAnalyzer(sim.model)
    freq_res = freq_analyzer.evaluate_frequency_grid(
        tx_pos, rx_pos, target_grid, fm_frequencies_hz
    )
    
    plt.figure(figsize=(15, 10))
    
    # (a) Individual FM Channels vs Angle (Zoomed 20° to 80°)
    plt.subplot(2, 2, 1)
    az_zoom_mask = (azimuth >= 20.0) & (azimuth <= 80.0)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, f_hz in enumerate(fm_frequencies_hz):
        snr_branch_db = 10.0 * np.log10(freq_res["snr_lin_matrix"][i, :])
        is_ref = (f_hz == fc_hz)
        plt.plot(
            azimuth[az_zoom_mask], snr_branch_db[az_zoom_mask],
            color=colors[i], lw=2.0 if is_ref else 1.2,
            linestyle="-" if is_ref else "--",
            label=f"{f_hz/1e6:.1f} MHz" + (" (Center)" if is_ref else "")
        )
    plt.title("FM Carrier Signals vs Angle (Zoomed $20^\\circ - 80^\\circ$)\nDemonstrating Fading Null Complementation across FM Band", fontsize=11, fontweight="bold")
    plt.xlabel("Azimuth Angle (deg)")
    plt.ylabel("Received SNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    
    # (b) Combining Performance (Single 98 MHz vs SC vs MRC across 5 FM channels)
    plt.subplot(2, 2, 2)
    single_98mhz_idx = 2  # 98 MHz is index 2
    snr_98mhz_db = 10.0 * np.log10(freq_res["snr_lin_matrix"][single_98mhz_idx, :])
    plt.plot(azimuth, snr_98mhz_db, "r-", lw=0.9, alpha=0.6, label="Single FM Station (98.0 MHz)")
    plt.plot(azimuth, freq_res["sc_snr_db"], "b-", lw=1.3, label="Selection Combining (SC, 5 FM Stations)")
    plt.plot(azimuth, freq_res["mrc_snr_db"], "g-", lw=1.5, label="Maximal Ratio Combining (MRC, 5 FM Stations)")
    plt.title("FM Band Diversity Combining Performance ($360^\\circ$ Coverage)", fontsize=11, fontweight="bold")
    plt.xlabel("Azimuth Angle (deg)")
    plt.ylabel("Received SNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    plt.legend(loc="lower right")
    
    # (c) Outage Probability / Fade CDF
    plt.subplot(2, 2, 3)
    snr_grid = np.linspace(-20, 30, 600)
    cdf_single = np.array([np.mean(snr_98mhz_db <= th) for th in snr_grid])
    cdf_sc = np.array([np.mean(freq_res["sc_snr_db"] <= th) for th in snr_grid])
    cdf_mrc = np.array([np.mean(freq_res["mrc_snr_db"] <= th) for th in snr_grid])
    
    plt.semilogy(snr_grid, np.maximum(cdf_single, 1e-4), "r-", lw=1.6, label="Single Station (98 MHz Baseline)")
    plt.semilogy(snr_grid, np.maximum(cdf_sc, 1e-4), "b-", lw=1.6, label="FM Frequency Diversity (SC, M=5)")
    plt.semilogy(snr_grid, np.maximum(cdf_mrc, 1e-4), "g-", lw=2.0, label="FM Frequency Diversity (MRC, M=5)")
    plt.axvline(0.0, color="k", linestyle=":", label="Threshold (0 dB)")
    plt.title("Fade Outage Probability CDF (88-108 MHz Band)", fontsize=11, fontweight="bold")
    plt.xlabel("SNR Threshold (dB)")
    plt.ylabel(r"Outage Probability $P(\mathrm{SNR} \leq \mathrm{Threshold})$")
    plt.ylim(1e-4, 1.0)
    plt.xlim(-15, 25)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right")
    
    # (d) Frequency Cross-Correlation Matrix
    plt.subplot(2, 2, 4)
    corr_f = freq_res["corr_matrix"]
    im_cf = plt.imshow(corr_f, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im_cf, label="Envelope Cross-Correlation")
    labels = [f"{f/1e6:.1f}M" for f in fm_frequencies_hz]
    plt.xticks(range(len(fm_frequencies_hz)), labels)
    plt.yticks(range(len(fm_frequencies_hz)), labels)
    for i in range(len(fm_frequencies_hz)):
        for j in range(len(fm_frequencies_hz)):
            plt.text(j, i, f"{corr_f[i, j]:.2f}", ha="center", va="center", color="black" if corr_f[i,j] < 0.65 else "white", fontweight="bold")
    plt.title("FM Frequency Cross-Correlation Matrix (88 - 108 MHz)", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    plot3_path = results_dir / "3_frequency_diversity_analysis.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")

    # =========================================================================
    # PART 4: SPATIAL / POSITION DIVERSITY AT 98 MHz
    # =========================================================================
    print("Running Spatial / Position Diversity Analysis at 98 MHz...")
    # At 98 MHz (lambda = 3.06m), mast height increments
    rx_heights_m = [10.0, 16.0, 24.0, 32.0]  # Receiver antenna heights on tower/mast
    rx_positions = [Position3D(rx_pos.x, rx_pos.y, h) for h in rx_heights_m]
    
    spatial_analyzer = SpatialDiversityAnalyzer(sim.model)
    spatial_res = spatial_analyzer.evaluate_rx_positions(
        tx_pos, rx_positions, target_grid
    )
    
    plt.figure(figsize=(15, 10))
    
    # (a) Individual Rx Heights vs Angle (Zoomed)
    plt.subplot(2, 2, 1)
    for i, h in enumerate(rx_heights_m):
        snr_branch_db = 10.0 * np.log10(spatial_res["snr_lin_matrix"][i, :])
        plt.plot(azimuth[az_zoom_mask], snr_branch_db[az_zoom_mask], lw=1.2, label=f"Rx Height {h:.0f} m")
    plt.title(f"Spatial Branch Signals vs Angle at 98 MHz (Zoomed $20^\\circ - 80^\\circ$)\nHeight Diversity ($h_r = 10, 16, 24, 32$ m)", fontsize=11, fontweight="bold")
    plt.xlabel("Azimuth Angle (deg)")
    plt.ylabel("Received SNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    
    # (b) Combining Performance
    plt.subplot(2, 2, 2)
    plt.plot(azimuth, spatial_res["single_branch_db"], "r-", lw=0.9, alpha=0.6, label="Single Antenna ($h_r = 10$ m)")
    plt.plot(azimuth, spatial_res["sc_snr_db"], "b-", lw=1.3, label="Position Diversity (SC, 4 Heights)")
    plt.plot(azimuth, spatial_res["mrc_snr_db"], "g-", lw=1.5, label="Position Diversity (MRC, 4 Heights)")
    plt.title("Spatial / Height Diversity Combining Performance (98 MHz)", fontsize=11, fontweight="bold")
    plt.xlabel("Azimuth Angle (deg)")
    plt.ylabel("Received SNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    plt.legend(loc="lower right")
    
    # (c) Outage Probability CDF
    plt.subplot(2, 2, 3)
    cdf_sp_single = np.array([np.mean(spatial_res["single_branch_db"] <= th) for th in snr_grid])
    cdf_sp_sc = np.array([np.mean(spatial_res["sc_snr_db"] <= th) for th in snr_grid])
    cdf_sp_mrc = np.array([np.mean(spatial_res["mrc_snr_db"] <= th) for th in snr_grid])
    
    plt.semilogy(snr_grid, np.maximum(cdf_sp_single, 1e-4), "r-", lw=1.6, label="Single Antenna Baseline ($h_r=10$m)")
    plt.semilogy(snr_grid, np.maximum(cdf_sp_sc, 1e-4), "b-", lw=1.6, label="Spatial Diversity (SC, M=4)")
    plt.semilogy(snr_grid, np.maximum(cdf_sp_mrc, 1e-4), "g-", lw=2.0, label="Spatial Diversity (MRC, M=4)")
    plt.axvline(0.0, color="k", linestyle=":", label="Threshold (0 dB)")
    plt.title("Spatial Diversity Outage CDF at 98 MHz", fontsize=11, fontweight="bold")
    plt.xlabel("SNR Threshold (dB)")
    plt.ylabel(r"Outage Probability $P(\mathrm{SNR} \leq \mathrm{Threshold})$")
    plt.ylim(1e-4, 1.0)
    plt.xlim(-15, 25)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right")
    
    # (d) Spatial Correlation Matrix
    plt.subplot(2, 2, 4)
    corr_sp = spatial_res["corr_matrix"]
    im_sp = plt.imshow(corr_sp, cmap="Purples", vmin=0, vmax=1)
    plt.colorbar(im_sp, label="Envelope Cross-Correlation")
    labels_sp = [f"{h:.0f}m" for h in rx_heights_m]
    plt.xticks(range(len(rx_heights_m)), labels_sp)
    plt.yticks(range(len(rx_heights_m)), labels_sp)
    for i in range(len(rx_heights_m)):
        for j in range(len(rx_heights_m)):
            plt.text(j, i, f"{corr_sp[i, j]:.2f}", ha="center", va="center", color="black" if corr_sp[i,j] < 0.65 else "white", fontweight="bold")
    plt.title("Spatial Cross-Correlation Matrix (Rx Heights)", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    plot4_path = results_dir / "4_spatial_diversity_analysis.png"
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved: {plot4_path}")

    # =========================================================================
    # PART 5: COMPARATIVE DIVERSITY SUMMARY
    # =========================================================================
    print("Generating Comparative Summary for 98 MHz & 88-108 MHz Band...")
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(azimuth, freq_res["div_gain_sc_db"], "b-", lw=1.1, alpha=0.8, label="FM Freq Diversity (SC Gain, 5 Stns)")
    plt.plot(azimuth, freq_res["div_gain_mrc_db"], "g-", lw=1.3, label="FM Freq Diversity (MRC Gain, 5 Stns)")
    plt.plot(azimuth, spatial_res["div_gain_mrc_db"], "m-.", lw=1.3, label="Spatial Diversity (MRC Gain, 4 Heights)")
    plt.title("Instantaneous Diversity Gain over Single 98 MHz Channel", fontsize=11, fontweight="bold")
    plt.xlabel("Azimuth Angle (deg)")
    plt.ylabel("Diversity Gain (dB)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    plt.legend(loc="upper right")
    
    plt.subplot(1, 2, 2)
    categories = [
        "Single 98MHz",
        "FM Div (SC, 5 Stns)",
        "FM Div (MRC, 5 Stns)",
        "Spatial Div (SC, 4 Hts)",
        "Spatial Div (MRC, 4 Hts)"
    ]
    p5_single = np.percentile(snr_98mhz_db, 5)
    p5_f_sc = np.percentile(freq_res["sc_snr_db"], 5)
    p5_f_mrc = np.percentile(freq_res["mrc_snr_db"], 5)
    p5_s_sc = np.percentile(spatial_res["sc_snr_db"], 5)
    p5_s_mrc = np.percentile(spatial_res["mrc_snr_db"], 5)
    
    mean_single = np.mean(snr_98mhz_db)
    mean_f_sc = np.mean(freq_res["sc_snr_db"])
    mean_f_mrc = np.mean(freq_res["mrc_snr_db"])
    mean_s_sc = np.mean(spatial_res["sc_snr_db"])
    mean_s_mrc = np.mean(spatial_res["mrc_snr_db"])
    
    x = np.arange(len(categories))
    width = 0.35
    plt.bar(x - width/2, [p5_single, p5_f_sc, p5_f_mrc, p5_s_sc, p5_s_mrc], width, label="5th Percentile SNR (Deep Fade Margin)", color="coral")
    plt.bar(x + width/2, [mean_single, mean_f_sc, mean_f_mrc, mean_s_sc, mean_s_mrc], width, label="Mean SNR across 360°", color="teal")
    plt.ylabel("SNR (dB)", fontsize=11)
    plt.title("Statistical Robustness Comparison (VHF FM Radar)", fontsize=11, fontweight="bold")
    plt.xticks(x, categories, rotation=15, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend(loc="upper left")
    
    plt.tight_layout()
    plot5_path = results_dir / "5_diversity_comparison_summary.png"
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"Saved: {plot5_path}")

    # Copy files to artifact dir
    artifact_dir = Path(r"C:\Users\m_mub\.gemini\antigravity\brain\2ad4a709-e895-4937-b6cd-fb0ad49bd9a2")
    for p in results_dir.glob("*.png"):
        dest = artifact_dir / p.name
        with open(p, "rb") as f_src, open(dest, "wb") as f_dst:
            f_dst.write(f_src.read())

    print("=" * 75)
    print("  FM-BAND (98 MHz, 88-108 MHz) SIMULATION METRICS")
    print("=" * 75)
    print(f"Single 98 MHz Mean SNR:              {mean_single:.2f} dB (5th %-tile Fade: {p5_single:.2f} dB)")
    print(f"FM Freq Diversity (SC, 5 Stns) Mean: {mean_f_sc:.2f} dB (5th %-tile: {p5_f_sc:.2f} dB, Null Gain: +{p5_f_sc - p5_single:.2f} dB)")
    print(f"FM Freq Diversity (MRC, 5 Stns) Mean:{mean_f_mrc:.2f} dB (5th %-tile: {p5_f_mrc:.2f} dB, Null Gain: +{p5_f_mrc - p5_single:.2f} dB)")
    print(f"Spatial Diversity (SC, 4 Hts) Mean:  {mean_s_sc:.2f} dB (5th %-tile: {p5_s_sc:.2f} dB, Null Gain: +{p5_s_sc - p5_single:.2f} dB)")
    print(f"Spatial Diversity (MRC, 4 Hts) Mean: {mean_s_mrc:.2f} dB (5th %-tile: {p5_s_mrc:.2f} dB, Null Gain: +{p5_s_mrc - p5_single:.2f} dB)")
    print("=" * 75)

if __name__ == "__main__":
    main()
