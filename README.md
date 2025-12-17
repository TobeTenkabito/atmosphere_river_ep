# Atmosphere_River_EP (Geo-Temporal Case-Crossover Modeling)

This repository contains the **Python implementation** for a **geo-temporal case-crossover model** used to quantify the robust and persistent relationship between **Atmospheric Rivers (ARs)** and **Extreme Precipitation (EP)** events across East Asia.

The methodology focuses on **causal inference** in a spatio-temporal context, utilizing advanced statistical techniques common in epidemiology and biostatistics.

## Technology Stack (Implemented Libraries)

This analysis leverages a high-performance scientific Python stack for efficient handling and processing of large-scale spatiotemporal data:

| **Category** | **Key Libraries** | **Usage** |
| :--- | :--- | :--- |
| **Data Ingestion & Alignment** | **`xarray`**, `netCDF4` | Core tools for reading, slicing, **resampling to daily frequency** (`time='D'`), and aligning massive NetCDF/GRIB datasets. |
| **Core Data Manipulation** | `NumPy`, `Pandas` | Efficient array computing, datetime indexing, and final data aggregation/CSV export. |
| **Performance & Parallelism** | **`Joblib`** | **Critical:** Implements multi-core parallel processing across the full spatial domain (all grid points) to drastically speed up diagnostics and statistical testing. |
| **Advanced Algorithms** | **`NumPy.fft`** | Used for efficient computation of Binary Cross-Correlation (via Fast Fourier Transform) for lag analysis. |
| **Statistical Modeling** | `SciPy.stats` (Implied) | Implementing core statistical tests (e.g., Binomial Test logic) and confidence interval calculations. |
| **Geospatial Visualization** | `Matplotlib`, `Cartopy` (Implied) | Generating publication-quality geographic maps, including custom color normalizations (`TwoSlopeNorm`) for correlation plots. |

## Repository Structure

The file naming convention reflects the project's complexity and iterative development:

* Files ending with `_v{x}.py` indicate standardized functions managing **statistical robustness checks** and minor variations in model parameters.
* Sequential scripts (`{number}step_file_name.py`) outline the end-to-end data processing and analysis workflow.

**Core Modules:**

* `ar_happen_calculate.py` and `precipitation_calculate.py`: Main modules for **data cleaning, event identification, and temporal matching**.
* **`diagnostics_module.py`:** Contains all time series analysis, lag selection, and sequential Monte Carlo simulation logic.
* Additional scripts provide supporting **statistical validation analyses**.

## Main Workflow & Data Analysis Highlights

### Input Data

* ERA5 reanalysis datasets (Climate Variables)
* Atmospheric River (AR) detection database

### Key Data Science Steps

**1. Data Wrangling and High-Performance Handling:**

* **Data Consistency:** Routines handle complex spatio-temporal alignment: reading data using `xarray` chunks, resampling to a common daily frequency, and ensuring inner-join alignment of time steps (`xr.align`).
* **Massive Parallelization:** **Crucial:** The `run_spatial_diagnostics` function uses **`Joblib`** to execute independent statistical tests concurrently for every single grid point in the spatial domain.

**2. Diagnostics and Optimal Lag Selection (Pre-Modeling):**

* **Lag Identification:** Employs multiple time series methods to identify the optimal lead-lag relationship (the time shift 'L' that maximizes the relationship):
    * **Event-Triggered Average (ETA):** E\[A\_{t-tau} | E\_t = 1\] for visual and quantitative lead-lag assessment.
    * **Binary Cross-Correlation:** Uses **FFT** for fast correlation calculation across a range of lags.
* **Adaptive Monte Carlo:** Implements a rigorous permutation test (`monthly_shuffle_test`) with **adaptive stopping rules** to ensure $p$-value stability while minimizing computational runtime.

**3. Core Case-Crossover Model Execution:**

* **Design Implementation:** Implements the **Case-Crossover Design** at the optimal lag ($L$) identified in the diagnostics phase. This quasi-experimental design inherently controls for time-invariant and slowly varying spatial/seasonal confounders.
* **Temporal Matching:** For each EP event, the model selects temporally adjacent **Control Periods** (e.g., matching the same time of day/year) to form the case/control pairs.
* **Core Calculation:** For each grid cell, the analysis is reduced to a $2\times2$ contingency table of **discordant pairs** (EP event under AR vs. EP event under non-AR control).

**4. Final Metrics Calculation & Interpretation:**

* **Causal Risk Quantification:** Calculates the **Pooled Odds Ratio (OR)** from the matched discordant pairs, which serves as the primary measure of association and causal risk (equivalent to Conditional Logistic Regression).
* **Attributable Fraction (AF & PAF):** Quantifies the ultimate impact metrics:
    * **Precipitation Efficiency (AF):** The ratio of AR-induced precipitation contribution to total precipitation.
    * **Population Attributable Fraction (PAF):** The percentage of EP events causally attributable to the AR exposure.
* **Statistical Validation:** The final results (OR, AF, PAF) are mapped spatially, utilizing the **monthly-shuffled $p$-values** and applying **Benjamini–Hochberg False Discovery Rate (FDR)** correction for robust significance testing across the entire domain.

## Environment & Disclaimer

* Developed and tested primarily in **Conda** environments, often utilized for managing complex scientific Python dependencies.
* **Disclaimer:** The full high-resolution datasets required for exact reproduction are proprietary and not included. The code is structured for **methodological review and demonstration** only, proving the functional implementation of the statistical framework.

**This code is provided for academic review purposes, highlighting the applicant's capability in advanced spatio-temporal data analysis, statistical modeling, and large-scale data processing.**
