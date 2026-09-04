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
    print("  SEPARATE RANGE AND ANGULAR FADING & DIVERSITY ANALYSIS (VHF FM BAND)")
    print("=" * 80)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fc_hz = 98.0e6
    tx_pos = Position3D(x=-20000.0, y=0.0, z=120.0)   # 120m FM broadcast mast at (-20km, 0)
    rx_pos = Position3D(x=20000.0, y=0.0, z=15.0)     # Surveillance mast at (+20km, 0)
    target_height_m = 2500.0                          # Target altitude 2.5 km
    
    sim = BistaticRadarSimulator(
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        frequency_hz=fc_hz,
        polarization=Polarization.VERTICAL,
        ground_type=GroundMedium.MEDIUM_GROUND,
        rms_roughness_m=0.05
    )
    
    # Diversity configurations
    fm_freqs_3 = np.array([88.5e6, 98.0e6, 107.5e6])
    rx_heights_2 = np.array([12.0, 22.0])
    
    # =========================================================================
    # PART 1: RANGE FADING PROFILES (SWEEPING RANGE FROM 5 km TO 100 km)
    # =========================================================================
    print("Running Radial Range Sweeps (5 km to 100 km)...")
    range_vec_m = np.linspace(5000.0, 100000.0, 1500)
    range_vec_km = range_vec_m / 1e3
    
    azimuths_eval = [90.0, 45.0, 0.0]
    az_names = ["Broadside ($90^\\circ$)", "Diagonal ($45^\\circ$)", "Endfire ($0^\\circ$)"]
    
    plt.figure(figsize=(16, 11))
    
    for plot_idx, az_deg in enumerate(azimuths_eval):
        az_rad = np.radians(az_deg)
        x_tgt = range_vec_m * np.cos(az_rad)
        y_tgt = range_vec_m * np.sin(az_rad)
        z_tgt = np.full_like(x_tgt, target_height_m)
        tgt_grid = np.column_stack((x_tgt, y_tgt, z_tgt))
        
        # 1. Single 98 MHz Baseline
        r_base = sim.model.compute(tx_pos, Position3D(20000.0, 0.0, 12.0), tgt_grid, frequency_hz=98.0e6)
        p_base = calculate_bistatic_received_power(r_base, frequency_hz=98.0e6)
        snr_base = p_base["snr_multipath_db"]
        snr_free = p_base["snr_freespace_db"]
        
        # 2. Freq Diversity Only (M=3)
        snr_f_lin = np.zeros((3, len(range_vec_m)))
        for i, f in enumerate(fm_freqs_3):
            r = sim.model.compute(tx_pos, Position3D(20000.0, 0.0, 12.0), tgt_grid, frequency_hz=f)
            p = calculate_bistatic_received_power(r, frequency_hz=f)
            snr_f_lin[i, :] = 10.0**(p["snr_multipath_db"] / 10.0)
        snr_f_mrc = 10.0 * np.log10(DiversityCombiner.maximal_ratio_combining(snr_f_lin))
        
        # 3. Height Diversity Only (M=2)
        snr_h_lin = np.zeros((2, len(range_vec_m)))
        for j, h in enumerate(rx_heights_2):
            r = sim.model.compute(tx_pos, Position3D(20000.0, 0.0, h), tgt_grid, frequency_hz=98.0e6)
            p = calculate_bistatic_received_power(r, frequency_hz=98.0e6)
            snr_h_lin[j, :] = 10.0**(p["snr_multipath_db"] / 10.0)
        snr_h_mrc = 10.0 * np.log10(DiversityCombiner.maximal_ratio_combining(snr_h_lin))
        
        # 4. Joint Hybrid Diversity (M = 3 freqs x 2 heights = 6)
        snr_hyb_lin = np.zeros((6, len(range_vec_m)))
        idx = 0
        for i, f in enumerate(fm_freqs_3):
            for j, h in enumerate(rx_heights_2):
                r = sim.model.compute(tx_pos, Position3D(20000.0, 0.0, h), tgt_grid, frequency_hz=f)
                p = calculate_bistatic_received_power(r, frequency_hz=f)
                snr_hyb_lin[idx, :] = 10.0**(p["snr_multipath_db"] / 10.0)
                idx += 1
        snr_hyb_sc = 10.0 * np.log10(DiversityCombiner.selection_combining(snr_hyb_lin))
        snr_hyb_mrc = 10.0 * np.log10(DiversityCombiner.maximal_ratio_combining(snr_hyb_lin))
        
        # Subplot for this Azimuth
        plt.subplot(2, 2, plot_idx + 1)
        plt.plot(range_vec_km, snr_base, "r-", lw=1.0, alpha=0.7, label="Single Baseline (98 MHz, 12m)")
        plt.plot(range_vec_km, snr_free, "k--", lw=1.2, alpha=0.6, label="Free-space SNR")
        plt.plot(range_vec_km, snr_f_mrc, "b-.", lw=1.2, label="Freq Diversity (MRC, 3 Freqs)")
        plt.plot(range_vec_km, snr_h_mrc, "m:", lw=1.3, label="Height Diversity (MRC, 2 Heights)")
        plt.plot(range_vec_km, snr_hyb_mrc, "g-", lw=2.0, label="Joint Hybrid (MRC, 6 Channels)")
        plt.title(f"Range Fading Profile: {az_names[plot_idx]}", fontsize=11, fontweight="bold")
        plt.xlabel("Bistatic Range from Midpoint (km)", fontsize=10)
        plt.ylabel("Received SNR (dB)", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(5, 100)
        plt.ylim(-25, 45)
        plt.legend(loc="upper right", fontsize=8)
        
    # (d) Range Diversity Outage CDF (aggregated over all ranges 5-100 km)
    plt.subplot(2, 2, 4)
    snr_grid = np.linspace(-20, 40, 600)
    cdf_r_base = np.array([np.mean(snr_base <= th) for th in snr_grid])
    cdf_r_f = np.array([np.mean(snr_f_mrc <= th) for th in snr_grid])
    cdf_r_h = np.array([np.mean(snr_h_mrc <= th) for th in snr_grid])
    cdf_r_hyb_sc = np.array([np.mean(snr_hyb_sc <= th) for th in snr_grid])
    cdf_r_hyb_mrc = np.array([np.mean(snr_hyb_mrc <= th) for th in snr_grid])
    
    plt.semilogy(snr_grid, np.maximum(cdf_r_base, 1e-4), "r--", lw=1.5, label="Single 98 MHz Baseline")
    plt.semilogy(snr_grid, np.maximum(cdf_r_h, 1e-4), "m:", lw=1.5, label="Height Diversity (MRC, M=2)")
    plt.semilogy(snr_grid, np.maximum(cdf_r_f, 1e-4), "b-.", lw=1.5, label="Freq Diversity (MRC, M=3)")
    plt.semilogy(snr_grid, np.maximum(cdf_r_hyb_sc, 1e-4), "c-", lw=1.8, label="Joint Hybrid (SC, M=6)")
    plt.semilogy(snr_grid, np.maximum(cdf_r_hyb_mrc, 1e-4), "g-", lw=2.2, label="Joint Hybrid (MRC, M=6)")
    plt.axvline(0.0, color="k", linestyle=":", label="0 dB Detection Floor")
    plt.title("Range Domain Outage Probability CDF (Broadside Cut)", fontsize=11, fontweight="bold")
    plt.xlabel("SNR Threshold (dB)", fontsize=10)
    plt.ylabel(r"Outage Probability $P(\mathrm{SNR} \leq \mathrm{Threshold})$", fontsize=10)
    plt.ylim(1e-4, 1.0)
    plt.xlim(-15, 35)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right", fontsize=8.5)
    
    plt.tight_layout()
    plot8_path = results_dir / "8_range_fading_and_diversity.png"
    plt.savefig(plot8_path, dpi=300)
    plt.close()
    print(f"Saved: {plot8_path}")

    # =========================================================================
    # PART 2: 2D POLAR RANGE-AZIMUTH COVERAGE MAP
    # =========================================================================
    print("Generating 2D Polar Range-Azimuth Map...")
    ranges_grid_km = np.linspace(5.0, 80.0, 200)
    az_grid_deg = np.linspace(0.0, 360.0, 360, endpoint=False)
    RR, AA = np.meshgrid(ranges_grid_km * 1e3, np.radians(az_grid_deg))
    
    XX = RR * np.cos(AA)
    YY = RR * np.sin(AA)
    ZZ = np.full_like(XX, target_height_m)
    grid_flat = np.column_stack((XX.ravel(), YY.ravel(), ZZ.ravel()))
    
    # 1. Single 98 MHz
    r_grid_base = sim.model.compute(tx_pos, Position3D(20000.0, 0.0, 12.0), grid_flat, frequency_hz=98.0e6)
    p_grid_base = calculate_bistatic_received_power(r_grid_base, frequency_hz=98.0e6)
    snr_base_2d = p_grid_base["snr_multipath_db"].reshape(XX.shape)
    
    # 2. Joint Hybrid (6 channels)
    snr_hyb_grid_lin = np.zeros((6, grid_flat.shape[0]))
    idx = 0
    for i, f in enumerate(fm_freqs_3):
        for j, h in enumerate(rx_heights_2):
            r = sim.model.compute(tx_pos, Position3D(20000.0, 0.0, h), grid_flat, frequency_hz=f)
            p = calculate_bistatic_received_power(r, frequency_hz=f)
            snr_hyb_grid_lin[idx, :] = 10.0**(p["snr_multipath_db"] / 10.0)
            idx += 1
    snr_hyb_mrc_flat = 10.0 * np.log10(DiversityCombiner.maximal_ratio_combining(snr_hyb_grid_lin))
    snr_hyb_mrc_2d = snr_hyb_mrc_flat.reshape(XX.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={"projection": "polar"})
    
    # Left: Single 98 MHz Polar Map
    ax_l = axes[0]
    im_l = ax_l.pcolormesh(AA, RR/1e3, snr_base_2d, cmap="jet", shading="auto", vmin=-10, vmax=35)
    cb_l = fig.colorbar(im_l, ax=ax_l, orientation="horizontal", pad=0.1)
    cb_l.set_label("Received SNR (dB)", fontsize=10)
    ax_l.set_title("Single Channel (98 MHz) Range-Azimuth SNR Map\n(Severe Range & Azimuth Blind Nulls)", fontsize=11, fontweight="bold", pad=15)
    ax_l.set_rticks([20, 40, 60, 80])
    ax_l.set_rlabel_position(45)
    
    # Right: Joint Hybrid Diversity (6 Channels MRC)
    ax_r = axes[1]
    im_r = ax_r.pcolormesh(AA, RR/1e3, snr_hyb_mrc_2d, cmap="jet", shading="auto", vmin=-10, vmax=35)
    cb_r = fig.colorbar(im_r, ax=ax_r, orientation="horizontal", pad=0.1)
    cb_r.set_label("Received SNR (dB)", fontsize=10)
    ax_r.set_title("Joint Hybrid (3 Freqs x 2 Heights = 6 MRC)\n(Complete Null Filling Across Range & Azimuth)", fontsize=11, fontweight="bold", pad=15)
    ax_r.set_rticks([20, 40, 60, 80])
    ax_r.set_rlabel_position(45)
    
    plt.tight_layout()
    plot9_path = results_dir / "9_range_azimuth_polar_map.png"
    plt.savefig(plot9_path, dpi=300)
    plt.close()
    print(f"Saved: {plot9_path}")
    
    # Copy to artifact dir
    artifact_dir = Path(r"C:\Users\m_mub\.gemini\antigravity\brain\2ad4a709-e895-4937-b6cd-fb0ad49bd9a2")
    for p in results_dir.glob("*.png"):
        dest = artifact_dir / p.name
        with open(p, "rb") as f_src, open(dest, "wb") as f_dst:
            f_dst.write(f_src.read())

    print("=" * 80)
    print("Range analysis completed successfully.")

if __name__ == "__main__":
    main()
