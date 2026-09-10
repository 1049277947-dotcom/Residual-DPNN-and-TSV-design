# TSV-design-Residual-DPNN-

This repository provides input scripts, sample data files, and a Python tool for building, training, and validating neural networks, including TSV optimization methods.

If you use this code, please cite:


## Contents

Note: In the filenames below, () indicates the specific structural layer (W, SiO2, or Si), and [] indicates the specific metal material (W, MO, M1-M7, or CU). Each placeholder corresponds to a separate file for that respective layer or material.

* `DATASET-().xlsx` — Summary data of maximum stresses for Each Layer
* `extract 20% text set.py` — Randomly splits data to create a 20% hold-out test set
* `extract-data.py` — Randomly samples 90% down to 7% of the data as training sets
* `Four-ablations-()-network bseline.py` — Trains the baseline NN
* `Four-ablations-()-predict bseline.py` — Evaluates predictions for the baseline NN
* `Four-ablations-()-network normal-residual.py` — Trains the "Normalized→residual" ablation network
* `Four-ablations-()-predict normal-residual.py` — Evaluates the "Normalized→residual" network
* `Four-ablations-()-network mech-all.py` — Trains the "Mech→full" ablation network
* `Four-ablations-()-predict mech-all.py` — Evaluates the "Mech→full" network
* `Four-ablations-()-network mech-residual.py` — Trains the proposed Residual-DPNNN
* `Four-ablations-()-predict mech-residual.py` — Evaluates the proposed Residual-DPNN
* `Dimensional-extroblation-().xlsx` — Test datasets for unseen geometric variables (depth and pitch)
* `Dimensional-extroblation-()-predict.py` — Tests geometric extrapolation of the frozen Residual-DPNN
* `Material-extroblation-[]-().xlsx` — Sparse supplementary datasets for new materials
* `Material-extroblation-()-Material-control-NN.py` — Trains the nested Material-Control NN
* `Material-extroblation-()-predict.py` — Tests stress predictions across different materials
* `Bayesian-optimizer.py` — Performs constrained Bayesian optimization for W-TSV design

## Requirements

* Python 3.x
