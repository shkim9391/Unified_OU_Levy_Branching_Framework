Unified Ornstein–Uhlenbeck–Lévy–Branching Framework

Overview

This repository implements the Unified Ornstein–Uhlenbeck–Lévy–Branching (OULB) Framework, a general probabilistic framework for modeling biological evolution that integrates

* continuous constrained dynamics (Ornstein–Uhlenbeck processes),
* abrupt evolutionary transitions (Lévy jump processes),
* lineage diversification (branching processes),
* heterogeneous observation models, and
* Bayesian statistical inference

within a single mathematical architecture.

Although the framework is applicable to many biological systems, this repository demonstrates its implementation using longitudinal pediatric leukemia data as the reference application.

⸻

Repository organization

The repository consists of two major components.

1. Reference application

application/pediatric_leukemia/

implements the complete five-stage analysis pipeline used in the accompanying manuscript.

2. Framework validation

figures/

contains simulation studies reproducing all methodological figures.

Computational workflow

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
Jump and branching inference

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
Figures 1–7

Analysis pipeline

⸻

Stage 1

Latent ecological projection

Main outputs

* projected latent states
* patient interval metrics
* frozen ecological scaffold

Directory

application/pediatric_leukemia/outputs/stage1/

⸻

Stage 2

Continuous evolutionary dynamics

This stage estimates

* OU attractors
* restoring strengths
* diffusion parameters
* model comparisons

Outputs include

* dynamic parameters
* Supplementary Figure S6
* regression validation

⸻

Stage 3

Discrete evolutionary events

This stage detects

* Lévy jump candidates
* branch switching
* ecological transitions
* escape risk

Outputs include

* relapse jump tables
* branch summaries
* threshold sensitivity analyses

⸻

Stage 4

Statistical validation

Produces

* effect sizes
* QQ analyses
* ranked displacement
* publication-ready Figure 3 tables

⸻

Stage 5

Simulation benchmarking

Evaluates

* parameter recovery
* jump detection accuracy
* branch recovery
* calibration to empirical data
* stress testing

⸻

Figures

The repository reproduces every figure in the manuscript.

Figure

Description

Figure 1

Unified OULB architecture

Figure 2

Simulation trajectories

Figure 3

Continuous parameter recovery

Figure 4

Recovery limits for jumps and branching

Figure 5

Observation-model robustness

Figure 6

Calibration to pediatric leukemia

Figure 7

Unified computational workflow

Running the framework

The complete reference analysis is executed sequentially.

bash application/pediatric_leukemia/run_stage1.sh

bash application/pediatric_leukemia/run_stage2.sh

bash application/pediatric_leukemia/run_stage3.sh

bash application/pediatric_leukemia/run_stage4.sh

bash application/pediatric_leukemia/run_stage5.sh

Each stage generates regression validation files, SHA256 checksums, and reproducible outputs.

⸻

Validation

Regression tests

python -m pytest

Integrity verification

shasum -a 256 *

ensures all published outputs match the released repository.

⸻

Mathematical framework

The latent state evolves according to

dX_t
=
\theta(\mu-X_t)\,dt
+
\sigma\,dW_t
+
dL_t
+
dB_t,

where

* \theta governs mean reversion,
* \mu is the attractor,
* dW_t denotes Brownian diffusion,
* dL_t represents Lévy jump processes,
* dB_t denotes branching events.

Observation models map latent trajectories to measured biological data while accounting for measurement noise and irregular sampling.

⸻

Reference application

The pediatric leukemia workflow demonstrates inference from longitudinal single-cell transcriptomic data through

1. ecological projection,
2. latent-state estimation,
3. continuous dynamics,
4. evolutionary jumps,
5. lineage branching,
6. simulation-based validation.

The same computational framework can be adapted to other biological systems by replacing the observation model while retaining the latent OULB process.

⸻

Citation

If you use this software, please cite:

Kim S-H. A Unified Ornstein–Uhlenbeck–Lévy–Branching Framework for Interpretable Modeling of Cancer Evolution. Mathematical and Computational Biology (in preparation).

