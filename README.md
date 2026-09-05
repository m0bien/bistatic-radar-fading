# Bistatic Radar Two-Ray Multipath Fading & 3D Joint Diversity Analysis Suite

An end-to-end, physics-accurate simulation engine and interactive HTML5 dashboard for analyzing **3D two-ray ground-reflection multipath fading**, **angular vs. range fading decoupling**, **3D multi-dimensional diversity (Tx Height &times; Rx Height &times; Frequency)**, and **statistical detection theory ($P_d, P_{\text{fa}}$, ROC, Outage CDF/CCDF)** in bistatic radar systems operating in the **VHF FM broadcast band (88–108 MHz)**.

---

## 📑 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Physics & Electromagnetic Formulation](#2-physics--electromagnetic-formulation)
   - [3D Vector Geometry & Image Theory](#21-3d-vector-geometry--image-theory)
   - [Complex Fresnel Reflection & Dielectric Ground](#22-complex-fresnel-reflection--dielectric-ground)
   - [Surface Roughness Model](#23-surface-roughness-model)
   - [Pattern Propagation Factors ($F_t, F_r, F^4$)](#24-pattern-propagation-factors-f_t-f_r-f4)
   - [Bistatic Radar Equation & Kell RCS Model](#25-bistatic-radar-equation--kell-rcs-model)
3. [Decoupled Fading Phenomena: Angle vs. Range](#3-decoupled-fading-phenomena-angle-vs-range)
   - [Angular Multipath Fading ($0^\circ - 360^\circ$)](#31-angular-multipath-fading-0---360)
   - [Radial Range Fading ($5 - 100\text{ km}$) & $R^{-8}$ Asymptotic Roll-Off](#32-radial-range-fading-5---100-km--r-8-asymptotic-roll-off)
4. [Multi-Dimensional Diversity Architecture](#4-multi-dimensional-diversity-architecture)
   - [Frequency Diversity (88–108 MHz FM Band)](#41-frequency-diversity-88108-mhz-fm-band)
   - [Tx Height Diversity & The "Illumination Shadow" Bottleneck](#42-tx-height-diversity--the-illumination-shadow-bottleneck)
   - [Pure Spatial $2\times 2$ MIMO Diversity](#43-pure-spatial-2x2-mimo-diversity)
   - [Joint 3D Hybrid Diversity Combiner ($M = M_f \times M_{s,t} \times M_{s,r}$)](#44-joint-3d-hybrid-diversity-combiner)
5. [Detection Theory & Statistical Metrics](#5-detection-theory--statistical-metrics)
   - [Neyman-Pearson Detection & The Fading Penalty](#51-neyman-pearson-detection--the-fading-penalty)
   - [Channel Hardening via Diversity](#52-channel-hardening-via-diversity)
   - [Outage Probability (CDF) vs. Reliability Coverage (CCDF)](#53-outage-probability-cdf-vs-reliability-coverage-ccdf)
6. [Interactive Dashboard Tab-by-Tab Guide](#6-interactive-dashboard-tab-by-tab-guide)
7. [Running the Codebase & Live Web Deployment](#7-running-the-codebase--live-web-deployment)

---

## 1. Executive Summary

Bistatic radars (such as Passive Bistatic Radar / PBR systems utilizing commercial FM radio illuminators of opportunity at $f_c \approx 98\text{ MHz}$, $\lambda \approx 3.06\text{ m}$) operate over terrestrial ground planes where multipath interference between the **direct line-of-sight (LOS) ray** and the **ground-reflected ray** produces severe destructive interference nulls ($> 25\text{ dB}$ signal dropouts).

This suite provides:
1. **Decoupled Analysis**: Separating **Angular Fading** (azimuthal interference lobes) from **Range Fading** (radial blind rings and long-range $R^{-8}$ ground cancellation).
2. **3D Joint Diversity Formulation**: Combining **Transmitter Height Diversity** ($M_{s,t}$), **Receiver Height Diversity** ($M_{s,r}$), and **FM Frequency Diversity** ($M_f$) into an $M$-branch Maximal Ratio Combiner (MRC).
3. **Detection Performance Verification**: Modeling instantaneous $P_d(\theta)$ coverage across $360^\circ$, false alarm rate $P_{\text{fa}}$, ROC curves, and Outage probability.

---

## 2. Physics & Electromagnetic Formulation

```
                 Target [x, y, z_t]
                     /\
       Direct       /  \      Direct
        Path       /    \      Path
       R_t,dir    /      \    R_r,dir
                 /        \
   Tx Tower     /          \     Rx Mast
  [x_t,y_t,h_t]/            \   [x_r,y_r,h_r]
      |       /              \       |
      |      /  Reflected     \      |
      |     /     Paths        \     |
══════╪════/════════════════════\════╪══════ Ground Plane (z = 0)
      |   /                      \   |
      |  /                        \  |
      | /                          \ |
   Image Tx                      Image Rx
 [x_t,y_t,-h_t]                [x_r,y_r,-h_r]
```

### 2.1 3D Vector Geometry & Image Theory
Let the transmitter, receiver, and target positions in 3D Cartesian coordinates be:

$$\mathbf{p}_t = \left[-\frac{L}{2}, 0, h_t\right]^T, \quad \mathbf{p}_r = \left[+\frac{L}{2}, 0, h_r\right]^T, \quad \mathbf{p}_{\text{tgt}} = [x, y, z_t]^T$$

where $L$ is the bistatic baseline distance, $h_t$ is the Tx mast height, $h_r$ is the Rx mast height, and $z_t$ is the target altitude above the flat ground plane ($z = 0$).

By electromagnetic image theory:
- **Tx-to-Target Direct Path**: $R_{td} = \|\mathbf{p}_{\text{tgt}} - \mathbf{p}_t\| = \sqrt{d_t^2 + (z_t - h_t)^2}$
- **Tx-to-Target Reflected Path**: $R_{tr} = \|\mathbf{p}_{\text{tgt}} - \mathbf{p}_t^{*}\| = \sqrt{d_t^2 + (z_t + h_t)^2}$
- **Tx Path Difference**: $\Delta R_t = R_{tr} - R_{td} \approx \frac{2 h_t z_t}{d_t}$
- **Tx Grazing Angle**: $\psi_t = \arctan\left(\frac{z_t + h_t}{d_t}\right)$

Similarly for the **Target-to-Rx Path**:
- **Target-to-Rx Direct Path**:
  
  $R_{rd} = \|-\mathbf{p}_{\text{tgt}} + \mathbf{p}_r\| = \sqrt{d_r^2 + (z_t - h_r)^2}$

- **Target-to-Rx Reflected Path**:

  $$R_{rr} = \|-\mathbf{p}_{\text{tgt}} + \mathbf{p}_r^*\| = \sqrt{d_r^2 + (z_t + h_r)^2}$$

- **Rx Path Difference**: $\Delta R_r = R_{rr} - R_{rd} \approx \frac{2 h_r z_t}{d_r}$
- **Rx Grazing Angle**: $\psi_r = \arctan\left(\frac{z_t + h_r}{d_r}\right)$

- **Note**: $(^*)$ is for the image path depicting the reflection. 
---

### 2.2 Complex Fresnel Reflection & Dielectric Ground
The ground is parameterized by relative permittivity $\epsilon_r$ and conductivity $\sigma$ (S/m). At angular frequency $\omega = 2\pi f$, the complex relative permittivity is:
$$\epsilon_c = \epsilon_r - j \frac{\sigma}{\omega \epsilon_0}$$

The complex Fresnel reflection coefficients $\Gamma(\psi)$ are given by:

#### Horizontal Polarization (TE / Perpendicular):
$$\Gamma_h(\psi) = \frac{\sin\psi - \sqrt{\epsilon_c - \cos^2\psi}}{\sin\psi + \sqrt{\epsilon_c - \cos^2\psi}}$$

#### Vertical Polarization (TM / Parallel):
$$\Gamma_v(\psi) = \frac{\epsilon_c \sin\psi - \sqrt{\epsilon_c - \cos^2\psi}}{\epsilon_c \sin\psi + \sqrt{\epsilon_c - \cos^2\psi}}$$

*Note: For vertical polarization, $\Gamma_v$ passes through a minimum at the pseudo-Brewster angle $\psi_B \approx \arcsin(1/\sqrt{\epsilon_r})$, causing a phase transition from $0^\circ$ to $-180^\circ$.*

---

### 2.3 Surface Roughness Model
Surface terrain irregularities reduce specular reflection. Using the **Miller-Brown / Ament** roughness reduction factor:
$$\rho_s = \exp\left(-\frac{1}{2} g^2\right), \quad g = \frac{4\pi \sigma_h \sin\psi}{\lambda}$$
where $\sigma_h$ is the terrain RMS roughness height and $\lambda = c / f$ is the radar wavelength.

---

### 2.4 Pattern Propagation Factors ($F_t, F_r, F^4$)
The complex voltage field propagation factors on the illumination and scatter paths are:
$$F_t = 1 + \Gamma_t \rho_{st} \exp\left(-j \frac{2\pi}{\lambda} \Delta R_t\right)$$
$$F_r = 1 + \Gamma_r \rho_{sr} \exp\left(-j \frac{2\pi}{\lambda} \Delta R_r\right)$$

The **Two-Way Bistatic Propagation Power Factor** is:
$$F^4 = |F_t|^2 \times |F_r|^2$$

- **Constructive Peak**: When $\Delta R \approx (n + 0.5)\lambda$, $F_t \to 2.0$, providing $|F|^2 = +6\text{ dB}$ on one link and up to **$+12\text{ dB}$** two-way gain.
- **Destructive Null**: When $\Delta R \approx n\lambda$, $F_t \to 0$, causing deep signal fades of **$-20\text{ dB}$ to $-40\text{ dB}$**.

---

### 2.5 Bistatic Radar Equation & Kell RCS Model
The received target echo power $P_r$ and SNR in multipath are:
$$P_r = \frac{P_t G_t G_r \lambda^2 \sigma_b}{(4\pi)^3 R_{td}^2 R_{rd}^2} \times F^4$$
$$\text{SNR} = \frac{P_r}{k_B T_0 B F_n}$$
where:
- $\sigma_b = \sigma_0 \cos^2(\beta / 2)$ is the bistatic target RCS (Kell model) as a function of bistatic angle $\beta$.
- $k_B T_0 B F_n$ is the receiver thermal noise floor.

---

## 3. Decoupled Fading Phenomena: Angle vs. Range

Multipath fading exhibits fundamentally decoupled behaviors along the **Angular** and **Radial Range** dimensions:

### 3.1 Angular Multipath Fading ($0^\circ - 360^\circ$)
- As a target flies an orbital path at constant range $R$, its relative distance to Tx ($d_t(\theta)$) and Rx ($d_r(\theta)$) varies continuously:
  $$d_t(\theta) = \sqrt{R^2 + (L/2)^2 + R L \cos\theta}, \quad d_r(\theta) = \sqrt{R^2 + (L/2)^2 - R L \cos\theta}$$
- This creates periodic angular lobes with spatial fringe spacing $\Delta \theta \approx \frac{\lambda R}{2 h z_t}$.
- In a single-channel radar, this produces sharp radial blind spokes where detection drops to zero.

### 3.2 Radial Range Fading ($5 - 100\text{ km}$) & $R^{-8}$ Asymptotic Roll-Off
- **Near/Intermediate Range ($5 - 35\text{ km}$)**: $\Delta R(R) \approx \frac{2 h z_t}{R}$ decreases with range, causing the signal to oscillate through multiple concentric **"blind null rings"**.
- **Far Range ($R > 50\text{ km}$)**: As $R \to \infty$, $\Delta R \to 0$ and $\psi \to 0$. Since $\Gamma \to -1$, the direct and reflected rays arrive almost $180^\circ$ out of phase:
  $$F_t \approx 1 - e^{-j k \Delta R_t} \approx j k \Delta R_t = j \frac{4\pi h_t z_t}{\lambda d_t} \propto \frac{1}{R}$$
- **The $R^{-8}$ Asymptotic Cancellation**: Substituting $F_t \propto 1/R$ and $F_r \propto 1/R$ into the radar equation yields:
  $$P_r \propto \frac{1}{R^4} \times |F_t|^2 |F_r|^2 \propto \frac{1}{R^4} \times \frac{1}{R^4} = \mathbf{\frac{1}{R^8}}$$
  This steep $R^{-8}$ roll-off severely restricts detection range unless diversity combines orthogonal lobes.

---

## 4. Multi-Dimensional Diversity Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               3D JOINT DIVERSITY MATRIX                │
                    │         M = M_f  x  M_{s,t}  x  M_{s,r} = 12           │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
               ┌───────────────────────────────┼──────────────────────────────┐
               ▼                               ▼                              ▼
      FREQUENCY DIVERSITY             TX HEIGHT DIVERSITY            RX HEIGHT DIVERSITY
   3 FM Carriers (M_f = 3)         Dual Mast Bays (M_{s,t} = 2)   Dual Mast Antennas (M_{s,r} = 2)
   f1 = 88.5 MHz                   h_{t1} = 120 m                 h_{r1} = 12 m
   f2 = 98.0 MHz                   h_{t2} = 180 m                 h_{r2} = 22 m
   f3 = 107.5 MHz                  (Fills Tx illumination nulls)  (Fills Rx scattering nulls)
```

### 4.1 Frequency Diversity (88–108 MHz FM Band)
Although the absolute bandwidth is $20\text{ MHz}$, the **fractional bandwidth is $20.4\%$**. 
The phase difference shift across the band is:
$$\Delta \phi = \frac{4\pi \Delta f}{c} \frac{h_t z_t}{d_t} \approx 300^\circ$$
This phase rotation transforms deep destructive nulls at $88.5\text{ MHz}$ into constructive peaks at $98.0\text{ MHz}$ or $107.5\text{ MHz}$.

### 4.2 Tx Height Diversity & The "Illumination Shadow" Bottleneck
Because $F^4 = |F_t|^2 \cdot |F_r|^2$, if the target falls into a Tx illumination null ($F_t \approx 0$), **no receiver diversity can detect the target because zero energy is scattered**. 
- Dual Tx antenna bays ($h_{t1} = 120\text{ m}, h_{t2} = 180\text{ m}$) decorrelate the illumination link $F_t$, providing **$+9.68\text{ dB}$** worst-case fade mitigation on its own.

### 4.3 Pure Spatial $2\times 2$ MIMO Diversity
By combining 2 Tx heights with 2 Rx heights at a single carrier frequency ($98\text{ MHz}$), we obtain $M = 4$ independent propagation channels ($T_1 R_1, T_1 R_2, T_2 R_1, T_2 R_2$) delivering **$+12.40\text{ dB}$ gain without requiring additional RF spectrum**.

### 4.4 Joint 3D Hybrid Diversity Combiner
Combining all 12 branches ($3\text{ Frequencies} \times 2\text{ Tx Heights} \times 2\text{ Rx Heights}$) using **Maximal Ratio Combining (MRC)**:
$$\gamma_{\text{MRC}} = \sum_{m=1}^{12} \gamma_m$$
- **5th-percentile SNR Floor**: Elevated from $-17.64\text{ dB}$ up to **$+3.07\text{ dB}$** (a **$+20.71\text{ dB}$ net gain**).
- **Outage Probability**: Reduced from $100\%$ down to **$0.00\%$**.

---

## 5. Detection Theory & Statistical Metrics

### 5.1 Neyman-Pearson Detection & The Fading Penalty
- In ideal AWGN (Marcum non-fluctuating target), achieving $P_d = 0.90$ at $P_{\text{fa}} = 10^{-6}$ requires an SNR of **$13.29\text{ dB}$**.
- In single-channel Rayleigh multipath fading (Swerling-I), $P_d = P_{\text{fa}}^{\frac{1}{1 + \bar{\gamma}}}$, requiring an average SNR of **$21.16\text{ dB}$**.
- **The Fading Penalty is $+7.87\text{ dB}$**.

### 5.2 Channel Hardening via Diversity
When $M$ independent diversity channels are combined via MRC, the sum SNR follows a **Gamma distribution (Chi-Square with $2M$ degrees of freedom)**:

$$P_d(M) = \exp\left(-\frac{\gamma_{\text{th}}}{1 + \bar{\gamma}_0}\right) \sum_{k=0}^{M-1} \frac{1}{k!} \left(\frac{\gamma_{\text{th}}}{1 + \bar{\gamma}_0}\right)^k$$

As $M \to 12$, the variance vanishes (**Channel Hardening**), and the required SNR drops to **$9.76\text{ dB}$** (saving **$11.40\text{ dB}$ of transmitter power** compared to single channel).

### 5.3 Outage Probability (CDF) vs. Reliability Coverage (CCDF)
Outage and Reliability are **exact mathematical complements**:
$$\mathbf{P_{\text{reliability}}(\gamma_{\text{th}}) = 1.0 - P_{\text{outage}}(\gamma_{\text{th}})}$$

```
100% ┤ ──────────────────────────┐                  100% ┤ ┌───────────────────────────
     │                           │ (Reliability     │  │                           (Outage
     │   RELIABILITY (CCDF)      │  Falls)          │  │     OUTAGE (CDF)           Rises)
 50% ┤   P(SNR ≥ γ_th)           │             50% ┤  │     P(SNR ≤ γ_th)           │
     │                           │                  │  │                           │
  0% ┤                           └───────           0% ┤ ───────┘
     └──────────────┬────────────────────           └──────────────┬───────────────────
                  Low Threshold  High Threshold                  Low Threshold  High Threshold
```

- **Outage CDF $P(\gamma \le \gamma_{\text{th}})$**: The probability of radar failure. Starts at $0\%$ and rises to $100\%$ as the demand $\gamma_{\text{th}}$ increases.
- **Reliability CCDF $P(\gamma \ge \gamma_{\text{th}})$**: The probability of detection success. Starts at $100\%$ and falls to $0\%$.

---

## 6. Interactive Dashboard Tab-by-Tab Guide

| Tab | Name | Theoretical Formulation | What Is Displayed |
| :---: | :--- | :--- | :--- |
| **1** | **Angular Fading ($0^\circ - 360^\circ$)** | Azimuth sweep at fixed radius $R = 45\text{ km}$, evaluating $F^4(\theta)$ and $\text{SNR}(\theta)$. | Compares Single Channel (red), Spatial MIMO (dashed purple), Freq MRC (blue), and 3D Hybrid MRC (green). Displays 5th-% SNR floor and Net Gain badges. |
| **2** | **Range Fading ($5 - 100\text{ km}$)** | Radial range sweep along a user-selected azimuth cut $\theta$. Models $1/R$ phase scaling and long-range $R^{-8}$ roll-off. | Shows deep multipath nulls and blind rings collapsing below the yellow dashed Free-Space $R^{-4}$ curve, and how 3D Hybrid MRC fills every range null. |
| **3** | **2D Spatial Map** | $200\text{ km} \times 200\text{ km}$ grid computation of bistatic two-way power factor $F^4(x, y)$. | Real-time 2D spatial heatmap with colorbar legend ($-30\text{ dB}$ deep blue to $+12\text{ dB}$ red), Tx site (yellow), Rx site (green), baseline, and orbit ring. |
| **4** | **Outage CDF / CCDF** | Statistical cumulative probability distribution over all $360^\circ$ azimuths. | Interactive radio toggle between Outage CDF $P(\text{SNR} \le \gamma)$ and Reliability CCDF $P(\text{SNR} \ge \gamma)$ across thresholds from $-15\text{ dB}$ to $+40\text{ dB}$. |
| **5** | **P<sub>d</sub> & ROC Curves** | Neyman-Pearson detection theory & Chi-square $2M$-DOF combiner integration. | Real-time $P_d(\theta)$ azimuth sweep with an interactive $P_{\text{fa}}$ slider ($10^{-8}$ to $10^{-2}$). Displays spatial coverage $P_d$ and fading penalty savings (+11.4 dB). |

---

## 7. Running the Codebase & Live Web Deployment

### 🌐 Live Web Browser (Interactive Dashboard)
- **Local**: Open `bistatic_radar_dashboard.html` or `index.html` in any web browser (Chrome, Firefox, Edge, Safari). No web server or internet connection required.
- **GitHub Pages**: Enabled by hosting `index.html` on the `main` branch [https://m0bien.github.io/bistatic-radar-fading/].

### 🐍 Running the Python Simulation Suite
Ensure Python 3.8+ is installed with `numpy`, `scipy`, and `matplotlib`:

```bash
# 1. Run full test suite
python tests/test_radar.py

# 2. Run 360° Angular Fading & Diversity Analysis
python run_analysis.py

# 3. Run Radial Range Fading & 2D Polar Map Analysis
python run_range_analysis.py

# 4. Run Joint Tx & Rx Height Diversity Analysis
python run_tx_rx_height_diversity.py

# 5. Run Probability of Detection (Pd) & ROC Analysis
python run_pd_pfa_analysis.py
```

All generated high-resolution figures are saved directly to the `results/` directory.
