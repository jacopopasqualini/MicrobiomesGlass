# Linking complex microbial interactions and dysbiosis through a disordered Lotka-–Volterra model

This repository contains the code used in:

**Pasqualini et al. (2025)**  
*Microbiomes Through the Looking Glass: Linking Species Interactions to Dysbiosis through a Disordered Lotka–Volterra Framework*  
*eLife*  
https://doi.org/10.7554/eLife.105948.2

---

## Overview

This codebase implements the inference and analysis pipeline for a **disordered generalized Lotka–Volterra (dgLV)** framework applied to cross-sectional gut microbiome data.  
It reproduces the analyses used to distinguish healthy and dysbiotic microbiome states through macroecological order parameters, interaction heterogeneity, and stability metrics derived from disordered systems theory.

---

## Repository Structure

Repository Structure

The core of the repository is organized under the workflow/ folder:

The dglv/ directory contains the implementation of the disordered generalized Lotka–Volterra model, including samplers (sampler.py), optimization utilities (opgd.py), and a sub-package of analysis tools (omico/) for model-specific analysis, fitting, plotting, session management, and table handling.

The inference/ directory holds routines for moment-matching inference and optimization, encapsulated in inference.py and associated helper functions in opgd.py.

A separate top-level omico/ directory houses general utilities for data analysis that are reused across workflows, such as analysis, fitting, plotting, session management, and table processing.

Finally, the notebooks/ directory includes the main scripts used to reproduce the paper’s results: get_final.py, which regenerates the final figures, and get_ops.py, which computes the macroecological order parameters.

---

## Requirements

Python ≥ 3.9 with standard scientific libraries:

`numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

---

## Usage

The main analyses and figures from the paper can be reproduced by running the scripts in:

workflow/notebooks/

yaml
Copy code

Input data should be provided as **relative abundance tables**, consistent with the compositional framework described in the manuscript.

---

## Citation

If you use this code, please cite:

> Pasqualini J., Maritan A., Rinaldo A., Facchin F., Savarino E., Altieri A., Suweis S (2025)  
> *Microbiomes Through the Looking Glass: Linking Species Interactions to Dysbiosis through a Disordered Lotka–Volterra Framework*  
> *eLife*. https://doi.org/10.7554/eLife.105948.2

---

## Contact

For questions or issues, please contact jacopo.pasqualini95@gmail.com
