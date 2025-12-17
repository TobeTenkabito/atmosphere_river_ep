# Atmosphere_River_EP (Geo-Temporal Case-Crossover Modeling)

This repository contains the **Python implementation** for a **geo-temporal case-crossover model** used to quantify the robust and persistent relationship between **Atmospheric Rivers (ARs)** and **Extreme Precipitation (EP)** events across East Asia.

The methodology focuses on **causal inference** in a spatio-temporal context, utilizing advanced statistical techniques common in epidemiology and biostatistics.

## Technology Stack (Implemented Libraries)

This analysis leverages a high-performance scientific Python stack to handle large-scale spatiotemporal data efficiently:

| **Category** | **Key Libraries** | **Usage** | 
 | ----- | ----- | ----- | 
| **Data Ingestion & Storage** | `netCDF4` | Reading and slicing large NetCDF datasets (ERA5 reanalysis). | 
| **Core Data Manipulation** | `NumPy`, `Pandas` | Efficient array computing, datetime handling, and data structuring. | 
| **Big Data / Memory** | `SciPy.sparse` (CSC Matrix) | Crucial for handling large, sparse (mostly zero) spatiotemporal grids of extreme events, significantly reducing memory footprint. | 
| **Statistical Modeling** | `SciPy.stats` | Implementing the core Case-Crossover inference logic (`binomtest` on discordant pairs) and calculating confidence intervals (`norm`). | 
| **Performance / Parallelism** | `Joblib` | Enabling multi-core parallel processing (`N_JOBS = -1`) for grid-cell level Case-Crossover and Fisher's Exact Tests, drastically cutting down computation time. | 
| **Geospatial Visualization** | `Matplotlib`, `Cartopy` | Generating publication-quality geographic maps, including custom colormaps, log-scaled ratio plots, and feature overlay (coastlines, borders). | 

## Repository Structure

The file naming convention reflects the project's complexity and iterative development:

* Files ending with `_v{x}.py` indicate standardized functions managing **statistical robustness checks** and minor variations in model parameters.

* Sequential scripts (`{number}step_file_name.py`) outline the end-to-end data processing and analysis workflow.

**Core Modules:**

* `ar_happen_calculate.py` and `precipitation_calculate.py`: Main modules for **data processing, event identification, and temporal matching**.

* `precipitation_output_v{x}.py`: An alternative script optimized for environments with limited memory/storage, demonstrating efficient **data output handling**.

* Additional scripts provide supporting **statistical validation analyses**.

## Main Workflow & Data Analysis Highlights

### Input Data

* ERA5 reanalysis datasets (Climate Variables)

* Atmospheric River (AR) detection database

### Key Data Science Steps

**1. High-Performance Big Data Handling:**

* **Memory Efficiency:** Utilizing **Compressed Sparse Column (CSC) matrices** (`scipy.sparse`) to represent the spatio-temporal AR and EP event flags. This method is vital for analyzing decades of high-resolution (0.25° x 0.25°, 6-hourly) data on standard computing resources.

* **Ingestion:** Using `netCDF4` for direct, efficient reading and slicing of large climate datasets.

**2. Advanced Sampling & Causal Inference:**

* Implements the **Case-Crossover Design**, a quasi-experimental method designed to control for time-invariant and slowly changing spatial/seasonal confounders.

* **Temporal Matching:** Implements event-based sampling using **Temporal Control Matching** (`all-control`) and **Random Sub-sampling** (`sample-control`).

**3. Statistical Modeling & Validation:**

* **Core Model:** Conditional Logistic Regression (implicitly implemented via **McNemar's Test** and the **Binomial Test** on discordant pairs for grid-cell level analysis).

* **Metrics:** Calculates and visualizes key Epidemiological and Predictive Metrics:

  * **Odds Ratio (OR):** Quantifies the effect size (risk multiplier).

  * **Lift Index:** Calculates the ratio of $P(E|AR) / P(E|\text{no } AR)$ from traditional contingency tables.

  * **Significance:** Applies **Benjamini–Hochberg False Discovery Rate (FDR)** control for robust p-value correction across thousands of grid cells.

**4. Robustness Checks (Placebo Tests):**

* Includes routines for running **Time-shifting** and **Month-shuffling** placebo experiments to demonstrate that the observed AR–EP relationship is genuine, not due to chance, temporal autocorrelation, or seasonal bias.

## Environment & Disclaimer

* Developed and tested primarily in **Conda** environments, often utilized for managing complex scientific Python dependencies.

* **Disclaimer:** The full high-resolution datasets required for exact reproduction are proprietary and not included. The code is structured for **methodological review and demonstration** only, proving the functional implementation of the statistical framework.

**This code is provided for academic review purposes, highlighting the applicant's capability in advanced spatio-temporal data analysis, statistical modeling, and large-scale data processing.**
