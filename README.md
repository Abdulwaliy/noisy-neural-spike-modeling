# Finding Spikes in Noisy Neural Data: Modeling, Simulation & Detection

A comprehensive computational neuroscience framework for analyzing, decomposing, modeling, and simulating continuous extracellular neural recordings contaminated by low-frequency baseline wander and high-frequency background noise.

This repository provides both empirical spike detection pipelines and a ground-truth generative model that decomposes extracellular signals into four biophysically grounded mathematical components:

1. **The Spike (Timing):** Homogeneous **Poisson point process** ($\lambda \approx 4.20\text{ Hz}$) with biophysical refractory lockout.
2. **The Drift (Baseline Wander):** Non-stationary baseline modeled via a **Sine function** ($f \approx 0.091\text{ Hz}$, $T \approx 11.0\text{ s}$).
3. **The Noise (Background Floor):** Uncorrelated thermal/electronic fluctuations modeled via a **Gaussian distribution** ($\mathcal{N}(0, 0.298^2)$).
4. **Single Spike (Action Potential Waveform):** Stereotyped extracellular depolarization upstroke followed by an **Exponential decay function** ($\tau \approx 2.85\text{ ms}$).

---

## Repository Structure

```text
├── spike_data.npz                         # Ground-truth reference recording (10,000 samples, 10.0 s, 1 kHz)
├── simulated_spike_data.npz               # Synthesized recording (Seed = 42, matching shapes and dtypes)
│
├── Simulated_Spike_Data_Modeling.ipynb    # Comprehensive master notebook integrating all 4 models & diagnostics
├── 1_Spike_Poisson_Modeling_new.ipynb     # Poisson timing, sine drift, noise ACF & paired background diagnostics
├── 2_Drift_Sine_Modeling.ipynb            # Dedicated baseline drift sine modeling notebook
├── 3_Noise_Gaussian_Modeling.ipynb        # Dedicated Gaussian background noise notebook
├── 4_Single_Spike_ExpDecay_Modeling.ipynb # Dedicated single spike exponential decay waveform notebook
│
├── Assignment_2A.ipynb                    # Assignment 2A working notebook (Parts 1, 2, and 3)
├── Assignment_2A_Standard.ipynb           # Standardized executed Assignment 2A notebook
├── Assignment_2A_Submission.pdf           # Concise 3-page worked solution report for Assignment 2A
├── Assignment_2B.ipynb                    # Assignment 2B notebook (Detector implementation, Sensitivity, NumPy benchmark)
├── Assignment_2B_Submission.pdf           # Concise 3-page worked solution report for Assignment 2B
│
├── figures/                               # Publication-quality diagnostic figures (sim_fig1 through sim_fig6)
├── requirements.txt                       # Python dependencies
└── README.md                              # Repository documentation
```

---

## Ground-Truth Mathematical Decomposition

Continuous extracellular voltage $V(t)$ is modeled as:
$$V(t) = b(t) + \epsilon(t) + \sum_{k=1}^N s_k(t - \tau_k)$$

Where:
* $b(t)$ is the low-frequency baseline drift.
* $\epsilon(t)$ is additive background noise.
* $\{\tau_k\}$ are discrete action potential timestamps.
* $s_k(\Delta t)$ is the stereotyped action potential voltage waveform.

### Parameter Extraction from `spike_data.npz`

Before simulating, empirical parameters were extracted from `spike_data.npz`:
* **Spike Detection:** Primary threshold $V_{\text{th}} = 3.0\text{ V}$ with contiguous sample grouping detected **42 discrete action potentials** ($\hat{\lambda} = 4.20\text{ Hz}$).
* **Spike Masking:** A protective temporal mask excluding $[-6, +14]$ samples ($-6\text{ ms}$ to $+14\text{ ms}$) around each spike peak retained **9,141 spike-free samples** and excluded 859 samples.
* **Sinusoidal Baseline Fit:** Non-linear least squares fit on unmasked samples yielded:
  $$b(t) = 0.6479 \sin(2\pi \cdot 0.0910 \cdot t + 0.2833) + 0.0044\text{ V}$$
  Producing a peak-to-peak baseline swing of $1.296\text{ V}$ across the $10.99\text{ s}$ period.
* **Gaussian Noise Estimation:** Residuals $e_i = y_i - \hat{b}(t_i)$ on unmasked samples yielded $\hat{\mu} = 0.0000\text{ V}$ and $\hat{\sigma} = 0.2983\text{ V}$ (`ddof=0`).
* **Exponential Decay Waveform:** Least-squares fit on the mean empirical spike repolarization phase ($\Delta t \ge 0$) yielded:
  $$V_{\text{spike}}(\Delta t) = 4.9550 \exp\left(-\frac{\Delta t}{0.00285}\right) + 0.1585\text{ V} \quad (R^2 = 0.9912)$$

---

## Summary of Reference vs. Simulated Data (`random_seed = 42`)

| Quantity / Property | Original Reference (`spike_data.npz`) | Simulated Data (`simulated_spike_data.npz`) | Agreement Assessment |
| :--- | :---: | :---: | :--- |
| **Array Shapes & Types** | `(10000,)`, `float64` | `(10000,)`, `float64` | **Identical** |
| **Sampling Rate ($f_s$) & Duration** | $1000.0\text{ Hz}$ ($10.0\text{ s}$) | $1000.0\text{ Hz}$ ($10.0\text{ s}$) | **Identical** |
| **Spike Event Count ($N$)** | 42 spikes | 47 spikes | Within $1\sigma$ Poisson spread ($\sqrt{42} \approx 6.5$) |
| **Average Firing Rate ($\lambda$)** | $4.20\text{ Hz}$ | $4.70\text{ Hz}$ | Within 95% Poisson CI $[2.93\text{ Hz}, 5.47\text{ Hz}]$ |
| **Mean Inter-Spike Interval (ISI)** | $234.1\text{ ms}$ | $208.5\text{ ms}$ | Conforms to exponential scale $1/\lambda$ |
| **ISI Coefficient of Variation ($CV$)** | $1.074$ | $0.985$ | Consistent with Poisson unity ($CV = 1.0$) |
| **Fano Factor ($F = s^2/\bar{N}$)** | $1.259$ | $1.082$ | Close to theoretical Poisson unity ($F = 1.0$) |
| **Drift Amplitude ($A$) & Frequency ($f$)** | $0.6479\text{ V}$, $0.0910\text{ Hz}$ | $0.6479\text{ V}$, $0.0910\text{ Hz}$ | Matched by construction |
| **Noise Std Dev ($\sigma$)** | $0.2983\text{ V}$ | $0.2979\text{ V}$ | Matched within $0.0004\text{ V}$ |
| **Total Signal Mean Voltage** | $+0.0769\text{ V}$ | $+0.0792\text{ V}$ | Matched within $0.0023\text{ V}$ |
| **Total Signal Standard Deviation** | $0.7344\text{ V}$ | $0.7358\text{ V}$ | Matched within $0.0014\text{ V}$ |
| **Detector Recovery at $V_{\text{th}} = 3.0\text{ V}$** | 42 spikes | 47 spikes | **100% Precision & 100% Recall** |

---

## Statistical & Biophysical Diagnostics

### 1. Autocorrelation Function (ACF) of Noise Residuals
To verify that the residual noise after sinusoidal baseline subtraction is white (temporally independent), the sample autocorrelation function was computed across 50 lags:
* **Lag-1 ACF:** $r_1 = -0.0200$, well within the 95% confidence bounds ($\pm 1.96 / \sqrt{N} = \pm 0.0206$).
* **Lag-20 ACF:** $r_{20} = 0.0058$ ($20\text{ ms}$ corresponds to the $50\text{ Hz}$ mains period at $1\text{ kHz}$), confirming no residual powerline hum in the reference recording.

### 2. Gaussian Normality Checks
* A normal Q-Q plot on residual noise yields $R^2 = 0.9998$.
* A two-sided Kolmogorov-Smirnov test against $\mathcal{N}(0, 0.2983^2)$ yields $p = 0.7511 \gg 0.05$.

### 3. Matched Background Drift + Noise Verification
Evaluating the combined background ($b(t) + \epsilon(t)$) at the exact 9,141 unmasked sample locations yields:
* Original background mean: $+0.0044\text{ V}$ vs. Simulated background mean: $+0.0009\text{ V}$.
* Original background standard deviation: $0.5471\text{ V}$ vs. Simulated background standard deviation: $0.5470\text{ V}$.

---

## Critical Scientific Caveats & Limitations

1. **The 95% Confidence Level ($\alpha = 0.05$) Pitfall:** Standard statistical significance ($V_{\text{th}} = F_V^{-1}(0.95) = 0.9953\text{ V}$) captures 500 samples across **279 false-positive noise triggers**. Because drifted noise peaks reach $\approx 2.20\text{ V}$, a valid spike threshold must be placed on the stable plateau ($[2.81\text{ V}, 4.08\text{ V}]$), such as $V_{\text{th}} = 3.0\text{ V}$.
2. **Global Threshold vs. Severe Baseline Wander:** A static threshold of $3.0\text{ V}$ succeeds on this recording because spike peaks ($>4.12\text{ V}$) remain separated from the drifted noise floor ($<2.4\text{ V}$). In recordings with larger baseline wander ($>3\text{ V}$), adaptive local thresholding or digital high-pass filtering is required.
3. **Microscopic Breakdown of Poisson Independence:** While macroscopic 1-second counts conform to Poisson statistics ($F \approx 1.0$, exponential ISI $p = 0.72$), the biophysical absolute refractory period ($\tau_{\text{ref}} \ge 2\text{--}5\text{ ms}$) strictly prevents sub-millisecond inter-spike intervals.
4. **Real-World 50 Hz Powerline Mains Hum (Nigerian Engineering Context):** In Nigeria, the 50 Hz / 230 V electrical grid and backup diesel/petrol generator or solar inverter harmonics couple $20\text{ ms}$ sinusoidal interference ($20\text{ samples}$ at $1\text{ kHz}$). Positive crests cause periodic false positives, and periodic clustering violates Poisson memorylessness unless filtered by a digital 50 Hz notch filter.
5. **Drift Non-Periodicity:** The sine function model assumes pure periodicity. Over extended recordings, biological electrode drift exhibits $1/f^\alpha$ pink noise or polarization ramps.

---

## Installation & Usage

### 1. Prerequisites
Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/username/noisy-neural-spike-modeling.git
cd noisy-neural-spike-modeling
pip install -r requirements.txt
```

### 2. Running the Notebooks
Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

* Open [`Simulated_Spike_Data_Modeling.ipynb`](Simulated_Spike_Data_Modeling.ipynb) to inspect the complete integrated modeling, synthesis, and comparative analysis pipeline.
* Open [`1_Spike_Poisson_Modeling_new.ipynb`](1_Spike_Poisson_Modeling_new.ipynb) to explore the expanded diagnostics for spike timing, sine drift fitting, and noise autocorrelation analysis.
* Open [`Assignment_2B.ipynb`](Assignment_2B.ipynb) to inspect the detector implementation, unit test suite, and the $\approx 71\times$ vectorized NumPy speedup benchmark.

### 3. Generating the Simulated Data via Python Script
To regenerate `simulated_spike_data.npz` and all diagnostic figures programmatically:

```bash
python generate_simulation_modeling.py
```

---

## License
MIT License. Free for educational, academic, and research purposes.
