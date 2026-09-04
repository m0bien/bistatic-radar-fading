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
from src.diversity import DiversityCombiner
from src.radar_equation import calculate_bistatic_received_power

def main():
    print("=" * 80)
    print("  HYBRID JOINT FREQUENCY + HEIGHT DIVERSITY ANALYSIS (VHF FM BAND)")
    print("=" * 80)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Base Parameters
    fc_hz = 98.0e6
    tx_pos = Position3D(x=-20000.0, y=0.0, z=120.0)   # 120m FM broadcast mast
    ground_type = GroundMedium.MEDIUM_GROUND
    polarization = Polarization.VERTICAL
    rms_roughness_m = 0.05
    
    # Practical Hybrid Configuration:
    # 3 FM Stations: Low, Mid, High of the FM band
    fm_frequencies_hz = np.array([88.5e6, 98.0e6, 107.5e6])  # M_f = 3
    # 2 Rx Mast Antennas: e.g., 12m and 22m (10m vertical baseline separation)
    rx_heights_m = np.array([12.0, 22.0])                     # M_s = 2
    
    # Comprehensive Grid for higher branch analysis:
    fm_freqs_5 = np.array([88.5e6, 93.0e6, 98.0e6, 103.0e6, 107.5e6]) # M_f = 5
    rx_heights_3 = np.array([12.0, 18.0, 26.0])                       # M_s = 3
    
    sim = BistaticRadarSimulator(
        tx_pos=tx_pos,
        rx_pos=Position3D(x=20000.0, y=0.0, z=12.0),
        frequency_hz=fc_hz,
        polarization=polarization,
        ground_type=ground_type,
        rms_roughness_m=rms_roughness_m
    )
    
    radius_m = 45000.0
    target_height_m = 2500.0
    num_angles = 1200
    
    azimuth_deg = np.linspace(0.0, 360.0, num_angles, endpoint=False)
    azimuth_rad = np.radians(azimuth_deg)
    x_tgt = radius_m * np.cos(azimuth_rad)
    y_tgt = radius_m * np.sin(azimuth_rad)
    z_tgt = np.full_like(x_tgt, target_height_m)
    target_grid = np.column_stack((x_tgt, y_tgt, z_tgt))
    
    # 1. Baseline: Single 98.0 MHz, Single 12m antenna
    res_base = sim.model.compute(tx_pos, Position3D(20000.0, 0.0, 12.0), target_grid, frequency_hz=98.0e6)
    pwr_base = calculate_bistatic_received_power(res_base, frequency_hz=98.0e6)
    snr_base_db = pwr_base["snr_multipath_db"]
    
    # 2. Freq-Only Diversity (M_f = 3 freqs, 1 height h=12m)
    snr_f_only_lin = np.zeros((len(fm_frequencies_hz), num_angles))
    for i, f in enumerate(fm_frequencies_hz):
        r = sim.model.compute(tx_pos, Position3D(20000.0, 0.0, 12.0), target_grid, frequency_hz=f)
        p = calculate_bistatic_received_power(r, frequency_hz=f)
        snr_f_only_lin[i, :] = 10.0**(p["snr_multipath_db"] / 10.0)
    snr_f_sc_db = 10.0 * np.log10(DiversityCombiner.selection_combining(snr_f_only_lin))
    snr_f_mrc_db = 10.0 * np.log10(DiversityCombiner.maximal_ratio_combining(snr_f_only_lin))
    
    # 3. Height-Only Diversity (M_s = 2 heights, 1 freq 98MHz)
    snr_h_only_lin = np.zeros((len(rx_heights_m), num_angles))
    for j, h in enumerate(rx_heights_m):
        r = sim.model.compute(tx_pos, Position3D(20000.0, 0.0, h), target_grid, frequency_hz=98.0e6)
        p = calculate_bistatic_received_power(r, frequency_hz=98.0e6)
        snr_h_only_lin[j, :] = 10.0**(p["snr_multipath_db"] / 10.0)
    snr_h_sc_db = 10.0 * np.log10(DiversityCombiner.selection_combining(snr_h_only_lin))
    snr_h_mrc_db = 10.0 * np.log10(DiversityCombiner.maximal_ratio_combining(snr_h_only_lin))
    
    # 4. Joint Hybrid Diversity (M = 3 freqs x 2 heights = 6 branches)
    m_hybrid_3x2 = len(fm_frequencies_hz) * len(rx_heights_m)
    snr_hybrid_3x2_lin = np.zeros((m_hybrid_3x2, num_angles))
    branch_labels_3x2 = []
    
    idx = 0
    for i, f in enumerate(fm_frequencies_hz):
        for j, h in enumerate(rx_heights_m):
            r = sim.model.compute(tx_pos, Position3D(20000.0, 0.0, h), target_grid, frequency_hz=f)
            p = calculate_bistatic_received_power(r, frequency_hz=f)
            snr_hybrid_3x2_lin[idx, :] = 10.0**(p["snr_multipath_db"] / 10.0)
            branch_labels_3x2.append(f"{f/1e6:.1f}M, h={h:.0f}m")
            idx += 1
            
    snr_hyb_3x2_sc_db = 10.0 * np.log10(DiversityCombiner.selection_combining(snr_hybrid_3x2_lin))
    snr_hyb_3x2_mrc_db = 10.0 * np.log10(DiversityCombiner.maximal_ratio_combining(snr_hybrid_3x2_lin))
    
    # 5. Ultra Hybrid Diversity (M = 5 freqs x 3 heights = 15 branches)
    m_hybrid_5x3 = len(fm_freqs_5) * len(rx_heights_3)
    snr_hybrid_5x3_lin = np.zeros((m_hybrid_5x3, num_angles))
    idx = 0
    for i, f in enumerate(fm_freqs_5):
        for j, h in enumerate(rx_heights_3):
            r = sim.model.compute(tx_pos, Position3D(20000.0, 0.0, h), target_grid, frequency_hz=f)
            p = calculate_bistatic_received_power(r, frequency_hz=f)
            snr_hybrid_5x3_lin[idx, :] = 10.0**(p["snr_multipath_db"] / 10.0)
            idx += 1
            
    snr_hyb_5x3_sc_db = 10.0 * np.log10(DiversityCombiner.selection_combining(snr_hybrid_5x3_lin))
    snr_hyb_5x3_mrc_db = 10.0 * np.log10(DiversityCombiner.maximal_ratio_combining(snr_hybrid_5x3_lin))
    
    # =========================================================================
    # PLOT 6: HYBRID JOINT DIVERSITY PERFORMANCE
    # =========================================================================
    plt.figure(figsize=(16, 11))
    
    # (a) Zoomed Angular Comparison (20° to 80°)
    plt.subplot(2, 2, 1)
    az_zoom = (azimuth_deg >= 20.0) & (azimuth_deg <= 80.0)
    plt.plot(azimuth_deg[az_zoom], snr_base_db[az_zoom], "r--", lw=1.2, alpha=0.7, label="Single Branch (98 MHz, h=12m)")
    plt.plot(azimuth_deg[az_zoom], snr_f_mrc_db[az_zoom], "b-.", lw=1.2, label="Freq Diversity (MRC, 3 Freqs)")
    plt.plot(azimuth_deg[az_zoom], snr_h_mrc_db[az_zoom], "m:", lw=1.4, label="Height Diversity (MRC, 2 Heights)")
    plt.plot(azimuth_deg[az_zoom], snr_hyb_3x2_sc_db[az_zoom], "c-", lw=1.5, label="Hybrid Freq+Height (SC, 6 Branches)")
    plt.plot(azimuth_deg[az_zoom], snr_hyb_3x2_mrc_db[az_zoom], "g-", lw=2.0, label="Hybrid Freq+Height (MRC, 6 Branches)")
    plt.title("Joint Hybrid vs Single Diversity (Zoomed $20^\\circ - 80^\\circ$)\nShowing Total Eradication of Fading Nulls", fontsize=11, fontweight="bold")
    plt.xlabel("Azimuth Angle (deg)")
    plt.ylabel("Received SNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    
    # (b) Full 360-Degree SNR Comparison
    plt.subplot(2, 2, 2)
    plt.plot(azimuth_deg, snr_base_db, "r-", lw=0.8, alpha=0.5, label="Single Baseline (98 MHz)")
    plt.plot(azimuth_deg, snr_hyb_3x2_sc_db, "c-", lw=1.2, label="Hybrid SC (3 Freqs x 2 Heights = 6)")
    plt.plot(azimuth_deg, snr_hyb_3x2_mrc_db, "g-", lw=1.6, label="Hybrid MRC (3 Freqs x 2 Heights = 6)")
    plt.plot(azimuth_deg, snr_hyb_5x3_mrc_db, "k-", lw=1.8, label="Ultra Hybrid MRC (5 Freqs x 3 Heights = 15)")
    plt.title("Full $360^\\circ$ Azimuth Coverage with Joint Diversity", fontsize=11, fontweight="bold")
    plt.xlabel("Azimuth Angle (deg)")
    plt.ylabel("Received SNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    plt.legend(loc="lower right", fontsize=9)
    
    # (c) Outage Probability CDF
    plt.subplot(2, 2, 3)
    snr_grid = np.linspace(-15, 35, 600)
    cdf_base = np.array([np.mean(snr_base_db <= th) for th in snr_grid])
    cdf_f = np.array([np.mean(snr_f_mrc_db <= th) for th in snr_grid])
    cdf_h = np.array([np.mean(snr_h_mrc_db <= th) for th in snr_grid])
    cdf_hyb_sc = np.array([np.mean(snr_hyb_3x2_sc_db <= th) for th in snr_grid])
    cdf_hyb_mrc = np.array([np.mean(snr_hyb_3x2_mrc_db <= th) for th in snr_grid])
    cdf_ultra_mrc = np.array([np.mean(snr_hyb_5x3_mrc_db <= th) for th in snr_grid])
    
    plt.semilogy(snr_grid, np.maximum(cdf_base, 1e-4), "r--", lw=1.5, label="Single Branch Baseline")
    plt.semilogy(snr_grid, np.maximum(cdf_h, 1e-4), "m:", lw=1.5, label="Height Only (MRC, M=2)")
    plt.semilogy(snr_grid, np.maximum(cdf_f, 1e-4), "b-.", lw=1.5, label="Freq Only (MRC, M=3)")
    plt.semilogy(snr_grid, np.maximum(cdf_hyb_sc, 1e-4), "c-", lw=1.8, label="Hybrid (SC, M=6)")
    plt.semilogy(snr_grid, np.maximum(cdf_hyb_mrc, 1e-4), "g-", lw=2.2, label="Hybrid (MRC, M=6)")
    plt.semilogy(snr_grid, np.maximum(cdf_ultra_mrc, 1e-4), "k-", lw=2.2, label="Ultra Hybrid (MRC, M=15)")
    plt.axvline(0.0, color="k", linestyle=":", label="0 dB Detection Floor")
    plt.title("Outage Probability CDF Comparison", fontsize=11, fontweight="bold")
    plt.xlabel("SNR Threshold (dB)")
    plt.ylabel(r"Outage Probability $P(\mathrm{SNR} \leq \mathrm{Threshold})$")
    plt.ylim(1e-4, 1.0)
    plt.xlim(-10, 30)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right", fontsize=8.5)
    
    # (d) 6x6 Hybrid Cross-Correlation Matrix
    plt.subplot(2, 2, 4)
    corr_6x6 = np.corrcoef(snr_hybrid_3x2_lin)
    im = plt.imshow(corr_6x6, cmap="magma_r", vmin=0, vmax=1)
    plt.colorbar(im, label="Cross-Correlation Coefficient")
    plt.xticks(range(m_hybrid_3x2), branch_labels_3x2, rotation=35, ha="right", fontsize=8)
    plt.yticks(range(m_hybrid_3x2), branch_labels_3x2, fontsize=8)
    for i in range(m_hybrid_3x2):
        for j in range(m_hybrid_3x2):
            plt.text(j, i, f"{corr_6x6[i, j]:.2f}", ha="center", va="center", color="black" if corr_6x6[i,j] < 0.6 else "white", fontsize=8, fontweight="bold")
    plt.title("Joint Freq-Height Cross-Correlation $\\rho(f_i, h_j)$", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    plot6_path = results_dir / "6_hybrid_joint_diversity_analysis.png"
    plt.savefig(plot6_path, dpi=300)
    plt.close()
    print(f"Saved: {plot6_path}")

    # Copy to artifact folder
    artifact_dir = Path(r"C:\Users\m_mub\.gemini\antigravity\brain\2ad4a709-e895-4937-b6cd-fb0ad49bd9a2")
    for p in results_dir.glob("*.png"):
        dest = artifact_dir / p.name
        with open(p, "rb") as f_src, open(dest, "wb") as f_dst:
            f_dst.write(f_src.read())

    # Calculate statistics
    p5_base = np.percentile(snr_base_db, 5)
    p5_f = np.percentile(snr_f_mrc_db, 5)
    p5_h = np.percentile(snr_h_mrc_db, 5)
    p5_hyb_sc = np.percentile(snr_hyb_3x2_sc_db, 5)
    p5_hyb_mrc = np.percentile(snr_hyb_3x2_mrc_db, 5)
    p5_ultra = np.percentile(snr_hyb_5x3_mrc_db, 5)
    
    m_base = np.mean(snr_base_db)
    m_f = np.mean(snr_f_mrc_db)
    m_h = np.mean(snr_h_mrc_db)
    m_hyb_sc = np.mean(snr_hyb_3x2_sc_db)
    m_hyb_mrc = np.mean(snr_hyb_3x2_mrc_db)
    m_ultra = np.mean(snr_hyb_5x3_mrc_db)

    print("=" * 80)
    print("  HYBRID JOINT DIVERSITY SIMULATION METRICS")
    print("=" * 80)
    print(f"Single Branch Baseline (98 MHz, 12m):  Mean = {m_base:.2f} dB | 5th %-tile = {p5_base:.2f} dB")
    print(f"Height Diversity Only (MRC, M=2):     Mean = {m_h:.2f} dB | 5th %-tile = {p5_h:.2f} dB  (Gain: +{p5_h - p5_base:.2f} dB)")
    print(f"Freq Diversity Only (MRC, M=3):       Mean = {m_f:.2f} dB | 5th %-tile = {p5_f:.2f} dB  (Gain: +{p5_f - p5_base:.2f} dB)")
    print(f"Hybrid Freq+Height (SC, M=3x2=6):     Mean = {m_hyb_sc:.2f} dB | 5th %-tile = {p5_hyb_sc:.2f} dB (Gain: +{p5_hyb_sc - p5_base:.2f} dB)")
    print(f"Hybrid Freq+Height (MRC, M=3x2=6):    Mean = {m_hyb_mrc:.2f} dB | 5th %-tile = {p5_hyb_mrc:.2f} dB (Gain: +{p5_hyb_mrc - p5_base:.2f} dB)")
    print(f"Ultra Hybrid (MRC, M=5x3=15):         Mean = {m_ultra:.2f} dB | 5th %-tile = {p5_ultra:.2f} dB (Gain: +{p5_ultra - p5_base:.2f} dB)")
    print("=" * 80)

if __name__ == "__main__":
    main()
