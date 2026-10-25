# $1 \neq 0$: A Thermodynamic Alternative to the Navier–Stokes Equations

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Preprint](https://img.shields.io/badge/Preprint-ChemRxiv-orange.svg)](https://chemrxiv.org)
[![Interactive Simulator](https://img.shields.io/badge/Live_Demo-GitHub_Pages-brightgreen.svg)](https://<your-username>.github.io/<your-repo-name>/fluid_dynamics_simulator.html)

Official companion code, benchmark suite, and analytical documentation for:
> **"$1 \neq 0$: A Thermodynamic Alternative to the Navier–Stokes Equations via Extended 14-Moment Hyperbolicity and Scale-Space Regularization"**  
> *Matthew B. Minus* (Department of Chemistry and Physics, Prairie View A&M University)[cite: 18]

---

## Overview

For nearly two centuries, the classical Navier–Stokes equations have modeled continuum fluid dynamics by projecting Newton's second law onto infinitesimal Eulerian control volumes[cite: 19]. Closing momentum balance with the algebraic Newton–Stokes constitutive relation ($\bm{\tau}^{\mathrm{dev}} = 2\mu \mathbf{D}^{\mathrm{dev}}$) enforces the Local Thermodynamic Equilibrium (LTE) truncation ($\tau_s \to 0$)[cite: 19, 21]. This singular limit creates infinite shear propagation speeds ($c_{\mathrm{shear}} \to \infty$) and permits artificial finite-time vorticity singularities ($\Vert{}\bm{\omega}\Vert{}_{L^\infty} \to \infty$) via unregularized vortex stretching[cite: 20].

This repository demonstrates that these pathologies are artifacts of replacing global thermodynamic functionals with instantaneous spatial derivative projections[cite: 19, 20]. Grounded in the **$1 \neq 0$ Non-Equivalence Principle** ($\tau_s > 0, \sigma > 0$)[cite: 21]:

1. **Strictly Symmetric Hyperbolic System:** The state space expands to a 14-moment Extended Irreversible Thermodynamic (EIT) manifold $\mathcal{M} = \{\rho, \mathbf{u}, e, \Pi, \bm{\tau}^{\mathrm{dev}}, \mathbf{q}\}$[cite: 21, 22]. Adopting the co-rotational Jaumann derivative $\overset{\circ}{(\cdot)}$ guarantees exact deviatoric trace preservation ($\mathrm{Tr}(\overset{\circ}{\bm{\tau}}^{\mathrm{dev}}) = 0$) and finite shear signal speeds $c_{\mathrm{shear}} = \sqrt{\mu / (\rho \tau_s)} < \infty$[cite: 21, 22, 23].
2. **Global $L^\infty$ Regularity (Theorem 5.1):** Continuum kinematics couple to a continuous Gaussian scale-space operator $\mathcal{S}_\lambda$ anchored at the minimum thermodynamic cutoff $\lambda_{\mathrm{diss}} = \sqrt{\nu \tau_s}$[cite: 23, 24]. Applying Young's convolution inequality across the global Helmholtz–Lyapunov availability functional $\mathcal{L}(t)$ proves that peak vorticity is unconditionally bounded ($\Vert{}\bm{\omega}_\lambda\Vert{}_{L^\infty} \le \Omega_{\max} < \infty$), satisfying the Beale–Kato–Majda (BKM) regularity criterion identically[cite: 23, 24].

---

## Repository Structure

```text
├── verify_1ne0_regularity.py              # Standalone Python verification suite (Lemma 3.1 & Theorem 5.1)
├── fluid_dynamics_simulator.html          # Interactive 2D real-time Eulerian vortex simulator
├── figure_1ne0_regularity_benchmark.pdf   # 3-panel publication benchmark graphic (Vector PDF)
├── figure_1ne0_regularity_benchmark.png   # 3-panel publication benchmark graphic (Raster PNG)
├── toc_graphic_1ne0.png                   # Table of Contents / Visual Abstract graphic
├── main.tex                               # Complete manuscript LaTeX source
├── LICENSE                                # Apache-2.0 License
└── README.md