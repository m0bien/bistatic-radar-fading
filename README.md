# Bistatic Radar Two-Ray Fading & Joint Diversity Analysis Suite

A physics-accurate Python simulation engine and interactive HTML5 GUI dashboard for analyzing 3D multipath ground-reflection fading and evaluating **Frequency Diversity (88–108 MHz FM band)**, **Height Diversity**, and **Joint Hybrid Diversity** in bistatic radar systems.

---

## 🌟 Key Features

1. **3D Vector Geometry & Image Theory**:
   - Exact modeling of direct paths ($R_{\text{td}}, R_{\text{rd}}$), ground-reflected paths ($R_{\text{tr}}, R_{\text{rr}}$), path differences ($\Delta R_t, \Delta R_r$), grazing angles ($\psi_t, \psi_r$), and bistatic angle $\beta$ for arbitrary positions $\text{Tx}(x_t, y_t, h_t)$, $\text{Rx}(x_r, y_r, h_r)$, and $\text{Target}(x, y, z)$.
2. **Dielectric Ground & Roughness Physics**:
   - Complex Fresnel reflection coefficients for Horizontal (TE) and Vertical (TM) polarizations across dielectric media (medium ground, wet ground, seawater, dry soil, PEC).
   - Miller-Brown / Ament surface roughness reduction factor.
3. **Decoupled Angle & Range Fading Analyses**:
   - **Angular Sweep ($0^\circ - 360^\circ$)**: Fading dynamics as the target rotates around the $\text{Tx}-\text{Rx}$ baseline.
   - **Range Sweeps ($5 - 100\text{ km}$)**: Radial range cuts showing lobing structure, blind null rings, and asymptotic $R^{-8}$ ground cancellation.
   - **2D Spatial Heatmaps & Polar Maps**: Full $(R, \theta)$ coverage disks.
4. **Diversity Combining Engine**:
   - Frequency Diversity across the FM broadcast band ($88.0 - 108.0\text{ MHz}$).
   - Spatial / Mast Height Diversity ($h_{r1}, h_{r2}$).
   - **Joint Hybrid Diversity ($M = M_f \times M_s$)**: Selection Combining (SC) and Maximal Ratio Combining (MRC).
5. **Interactive Real-Time HTML5 GUI Dashboard**:
   - Real-time parameter sliders (frequencies, mast heights, baseline, target altitude, orbit radius, range cut azimuth, polarization, ground type).
   - Instant live canvas plotting of angular fading, range fading, 2D heatmaps with colorbar legends, and Outage CDF / Coverage CCDF toggles.

---

## 📁 Repository Structure

```
├── bistatic_radar_dashboard.html   # Interactive HTML5/Canvas GUI Dashboard
├── src/
│   ├── geometry.py                 # 3D vector geometry, reflection points, path lengths
│   ├── reflection.py               # Fresnel reflection coefficients & roughness models
│   ├── propagation.py              # Two-ray pattern propagation factors (Ft, Fr, F^4)
│   ├── radar_equation.py           # Bistatic radar equation, received power, SNR, RCS
│   ├── diversity.py                # Frequency, Spatial, and Hybrid Diversity combiners (MRC, SC)
│   └── simulator.py                # Simulation coordinator for angular and spatial grids
├── tests/
│   └── test_radar.py               # Unit test suite
├── results/                        # Generated high-resolution simulation plots
├── run_analysis.py                 # 360° Angular fading & diversity analysis runner
├── run_range_analysis.py           # Radial range fading & 2D polar map runner
├── run_hybrid_diversity.py         # Joint hybrid (6-branch & 15-branch) diversity runner
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Matplotlib

```bash
pip install numpy scipy matplotlib
```

### 1. Run Unit Tests
```bash
python tests/test_radar.py
```

### 2. Run Comprehensive Simulations
```bash
# Run 360-degree angular fading & FM-band diversity
python run_analysis.py

# Run radial range sweeps and 2D polar map
python run_range_analysis.py

# Run joint hybrid frequency + height diversity comparison
python run_hybrid_diversity.py
```

### 3. Launch Interactive Dashboard
Double click `bistatic_radar_dashboard.html` or open it in any modern browser:
```powershell
Start-Process bistatic_radar_dashboard.html
```

---

## 📊 Summary of Results

| Architecture | Diversity Channels ($M$) | Mean Received SNR | 5th %-tile SNR (Deep Null Floor) | Worst-Case Fade Mitigation Gain | Outage Prob ($P_{\text{out}} \le 0\text{ dB}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Single Channel Baseline** ($98\text{ MHz}, 12\text{m}$) | $1$ | $12.19\text{ dB}$ | $+2.36\text{ dB}$ | Baseline ($0\text{ dB}$) | $5.1\%$ |
| **Height Diversity Only (MRC)** ($h = 12, 22\text{m}$) | $2$ | $14.85\text{ dB}$ | $+6.49\text{ dB}$ | $+4.13\text{ dB}$ | $0.8\%$ |
| **Freq Diversity Only (MRC)** ($88.5, 98, 107.5\text{M}$) | $3$ | $19.18\text{ dB}$ | $+17.51\text{ dB}$ | $+15.15\text{ dB}$ | $< 0.05\%$ |
| **Joint Hybrid Diversity (SC)** ($3\text{ Freqs} \times 2\text{ Heights}$) | $6$ | $17.36\text{ dB}$ | $+15.57\text{ dB}$ | $+13.21\text{ dB}$ | $\approx 0\%$ |
| **Joint Hybrid Diversity (MRC)** ($3\text{ Freqs} \times 2\text{ Heights}$) | **$6$** | **$21.90\text{ dB}$** | **$+19.79\text{ dB}$** | **$+17.43\text{ dB}$** | **$\approx 0\%$** |
| **Ultra Hybrid Diversity (MRC)** ($5\text{ Freqs} \times 3\text{ Heights}$) | **$15$** | **$25.50\text{ dB}$** | **$+24.25\text{ dB}$** | **$+21.89\text{ dB}$** | **$\approx 0\%$** |

---

## 📄 License
MIT License.
