# Unified Ornstein–Uhlenbeck–Lévy–Branching (OULB) Framework

> A unified probabilistic framework for interpretable modeling of biological evolution through continuous dynamics, Lévy jump processes, branching evolution, and Bayesian inference.

---

## Overview

The **Unified Ornstein–Uhlenbeck–Lévy–Branching (OULB) Framework** provides a general mathematical and computational framework for modeling biological evolution across multiple spatial and temporal scales. Rather than treating continuous dynamics, abrupt evolutionary transitions, lineage diversification, and measurement uncertainty as separate problems, OULB integrates these mechanisms within a single latent-state representation.

The framework combines

- **continuous constrained dynamics** using Ornstein–Uhlenbeck (OU) processes,
- **abrupt evolutionary transitions** using Lévy-like jump processes,
- **lineage diversification** through branching processes,
- **flexible observation models** for heterogeneous biological measurements, and
- **Bayesian statistical inference** for parameter estimation and uncertainty quantification.

Although applicable to many biological systems, this repository demonstrates the framework using **longitudinal pediatric leukemia** as a reference application.

---

# Repository Organization

The repository is organized into three primary components.

```
application/
```

Reference biological applications implementing the complete OULB inference workflow.

```
figures/
```

Simulation studies and scripts reproducing every figure in the accompanying manuscript.

```
src/
```

Core implementation of the reusable OULB framework.

Additional directories include

```
configs/
tests/
docs/
data/
```

for configuration files, regression testing, documentation, and datasets.

---

# Computational Workflow

```
Raw observations
        │
        ▼
Stage 1
Latent ecological projection
        │
        ▼
Stage 2
Continuous OU parameter estimation
        │
        ▼
Stage 3
Lévy jump and branching inference
        │
        ▼
Stage 4
Statistical validation
        │
        ▼
Stage 5
Simulation benchmarking
        │
        ▼
Publication Figures
        │
        ▼
Biological interpretation
```

---

# Analysis Pipeline

## Stage 1 — Latent Ecological Projection

Projects biological observations into a common latent scaffold.

**Primary outputs**

- projected latent states
- patient interval metrics
- frozen ecological scaffold

```
application/pediatric_leukemia/outputs/stage1/
```

---

## Stage 2 — Continuous Evolutionary Dynamics

Estimates continuous stochastic dynamics using Ornstein–Uhlenbeck models.

**Estimated quantities**

- attractor locations
- restoring strengths
- diffusion parameters
- model comparison statistics

**Outputs**

- dynamic parameter estimates
- Supplementary Figure S6
- regression validation reports

---

## Stage 3 — Lévy Jump and Branching Inference

Identifies discontinuous evolutionary events and lineage diversification.

**Detected events**

- Lévy jump candidates
- branch transitions
- ecological state switching
- evolutionary escape risk

**Outputs**

- relapse jump tables
- branch summaries
- threshold sensitivity analyses

---

## Stage 4 — Statistical Validation

Performs statistical evaluation of inferred evolutionary dynamics.

Outputs include

- effect sizes
- QQ analyses
- ranked displacement statistics
- publication-ready Figure 3 summary tables

---

## Stage 5 — Simulation Benchmarking

Evaluates the statistical performance of the framework through simulation.

Benchmark analyses include

- parameter recovery
- jump detection accuracy
- branch recovery
- calibration to empirical data
- stress testing
- estimator comparison

---

# Manuscript Figures

The repository reproduces every figure included in the accompanying methodological manuscript.

| Figure | Description |
|---------|-------------|
| **Figure 1** | Unified OULB architecture |
| **Figure 2** | Representative simulation trajectories |
| **Figure 3** | Continuous parameter recovery |
| **Figure 4** | Recovery limits for jump and branching inference |
| **Figure 5** | Observation-model robustness |
| **Figure 6** | Calibration to pediatric leukemia data |
| **Figure 7** | Unified computational workflow |

---

# Running the Framework

The complete reference analysis is executed sequentially.

```bash
bash application/pediatric_leukemia/run_stage1.sh

bash application/pediatric_leukemia/run_stage2.sh

bash application/pediatric_leukemia/run_stage3.sh

bash application/pediatric_leukemia/run_stage4.sh

bash application/pediatric_leukemia/run_stage5.sh
```

Each stage produces

- reproducible intermediate outputs
- regression validation reports
- SHA256 integrity checks
- publication-ready figures and tables

---

# Validation

Run the complete regression test suite

```bash
python -m pytest
```

Verify output integrity

```bash
shasum -a 256 *
```

These procedures ensure that all released outputs exactly match the archived repository version.

---

# Mathematical Framework

The latent biological state evolves according to

$begin:math:display$
dX\_t
\=
\\theta\(\\mu\-X\_t\)\\\,dt
\+
\\sigma\\\,dW\_t
\+
dL\_t
\+
dB\_t\,
$end:math:display$

where

| Symbol | Interpretation |
|---------|----------------|
| $\theta$ | Mean-reversion strength |
| $\mu$ | Evolutionary attractor |
| $dW_t$ | Brownian diffusion |
| $dL_t$ | Lévy jump process |
| $dB_t$ | Branching process |

Observation models map latent trajectories to measured biological observations while explicitly accounting for measurement uncertainty, irregular sampling, and heterogeneous experimental technologies.

---

# Reference Application

The reference implementation analyzes longitudinal pediatric leukemia through

1. latent ecological projection,
2. continuous evolutionary dynamics,
3. Lévy jump inference,
4. lineage branching,
5. Bayesian parameter estimation,
6. simulation-based validation.

The observation model can be replaced to accommodate other biological systems while retaining the same latent OULB process, making the framework broadly applicable to developmental biology, microbial evolution, ecology, and phylogenetic comparative analyses.

---

# Citation

If you use this software in your research, please cite

> **Kim S-H.** *A Unified Ornstein–Uhlenbeck–Lévy–Branching Framework for Interpretable Modeling of Cancer Evolution.* *Mathematical and Computational Biology* (in preparation). 

---

# License

This repository is distributed under the MIT License.

# Zenodo DOI

https://doi.org/10.5281/zenodo.21827910

---

# Acknowledgments

Development of the Unified OULB Framework was motivated by methodological challenges in modeling stochastic biological evolution across continuous, discontinuous, and branching processes. The pediatric leukemia application serves as the reference implementation demonstrating the flexibility and reproducibility of the framework.
