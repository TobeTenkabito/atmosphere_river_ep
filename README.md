# Atmosphere_River_EP

This repository contains code developed to demonstrate the relationship between **Atmospheric Rivers (ARs)** and **Extreme Precipitation (EP)** events.

---

## Repository Structure
- Files ending with `_v{x}.py` indicate different stable versions of the same function with minor variations.  
- `{number}step_file_name.py` scripts represent sequential steps of the workflow (follow the numbering order to process data).  
- `ar_happen_calculate.py` and `precipitation_calculate.py` are the core scripts.  
- `precipitation_output_v{x}.py` is an alternative for machines with limited storage or Windows systems.  
- Additional Python scripts provide supporting statistical analyses.  

---

## Main Workflow
1. **Input Data**  
   - ERA5 reanalysis datasets  
   - Atmospheric River (AR) detection datasets  

2. **Run Core Scripts**  
   - `ar_happen_calculate.py`  
   - `precipitation_calculate.py`  

3. **Optional**  
   - For storage-limited or Windows systems: run `precipitation_output_v{x}.py` after `precipitation_calculate.py`.  

4. **Key Steps**  
   - **Step 1 & Step 4 (main functions):**  
     - Handle high-resolution spatiotemporal reanalysis datasets (e.g., 0.25° × 0.25°, 6-hourly).  
     - Event-based sampling with **all-control** and **sample-control** modes.  
     - Statistical indicators:  
       - Odds Ratio (OR)  
       - Attributable Fraction (AF)  
       - Population Attributable Fraction (PAF)  
       - Lift Index  
     - Grid-based and regional aggregation.  
     - Global and regional domain analysis.  
     - Visualization of results.  

   - **Step 2 & Step 3:** exploratory analyses (not central to the framework).  

---

## Environment
- Developed and tested in **Conda** environments.  
- Functionality on other environments/operating systems is **not guaranteed**.  
- Scripts have been primarily tested on **Windows OS**.  

---

## Disclaimer
This repository contains code developed during **ongoing research**.  
- The datasets used in the experiments are **not publicly available**.  
- Results reported in the related manuscript **cannot be fully reproduced** with this code alone.  
- This code is provided **for demonstration and academic review purposes only**.  

---

## Citation
If you use this code for academic purposes, please cite the related manuscript once it becomes available.
