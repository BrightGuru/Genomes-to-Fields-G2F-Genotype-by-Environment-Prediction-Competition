# Genomes to Fields (G2F)
Genotype by Environment Prediction Competition 2024

Using an ensemble of classical and Machine learning methods in genomic prediction in multiple environments.

## Code Structure and Script Description

This repository contains scripts for phenotypic preprocessing, genomic data integration, and model development for maize yield prediction using gradient boosting and ensemble learning.

### Data Preprocessing and Preliminary Analysis

**Phenotypic and Genotypic Preliminary Analysis**
- *Preliminary_Analysis.R*
- Performs data cleaning, outlier detection, BLUE estimation, and initial exploratory analyses for phenotypic and genotypic datasets. It also contains code for creating additive-centered (Z) and dominance deviation (W) matrices

### Model Implementations

1. XGBoost Model (Concatenated Features) - *G2F_normal.py*
- Trains an XGBoost model using a fully concatenated feature matrix comprising SNP markers, metadata, and weather variables.

2. 2NP Weighted Ensemble (Additive and Dominance)
- *G2F_addev.py*
- *G2F_domdev.py*
- Implements the 2NP ensemble framework by training separate XGBoost models on the additive-centered (Z) and dominance deviation (W) matrices, each concatenated with metadata and weather data. Predictions are combined using variance-based weighting of additive and dominance components.

3. 2NP Weighted Ensemble (Additive, Dominance, and Environmental)
- *G2F_addev.py*
- *G2F_domdev.py*
- *G2F_meta_weather.py*
- Extends the 2NP ensemble by including an additional model trained exclusively on metadata and weather variables. Predictions from additive, dominance, and environmental models are combined using weights derived from additive, dominance, and residual variance components.

4. 2NP Ensemble with Equal Weights
- *G2F_addev.py*
- *G2F_domdev.py*
- Combines predictions from additive and dominance XGBoost models using equal weighting, without variance-based scaling.

5. XGBoost and LightGBM Ensemble
- *G2F_normal.py*
- *G2F_LGBM_normal.py*
- Trains parallel XGBoost and LightGBM models on the same concatenated feature matrix (SNP markers, metadata, and weather data). Final predictions are obtained via simple averaging of model outputs.
