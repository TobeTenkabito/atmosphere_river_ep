# atmosphere_river_ep
This is a project to demonstrate the relationship between atmosphere rivers(ARs) and extreme precipitation(EP).
The end "name _v{x}.py" means different versions;they are slightly different in functionality and details,but all of these versions are stable.
All the code is used in Windows,the other operate systerm is not tested yet.

How to use?
First input your dataset ERA5 reanalysis and atmospheric rivers data and run the "ar_happen_calculate.py" and "precipitation_calculate.py";if your computer has no enough space or your operate systerm is windows，then you can run the "precipitation_output_v{x}.py" after runing "precipitation_calculate.py" file.
Find the file name as "{number}step_file_name.py" and follow the number to calculate these data and know how we demonstrate results.
The other statistic information is the rest python file.

And this code is tested on the conda;it is unknown whether it can run successfully on the other environment.

The main function is step1 and step4:
- Handles high-resolution spatiotemporal reanalysis datasets (e.g., 0.25° × 0.25°, 6-hourly).
- Event-based sampling framework with both **all-control** and **sample-control** modes.
- Computation of statistical indicators such as:
  - Odds Ratio (OR)
  - Attributable Fraction (AF)
  - Population Attributable Fraction (PAF)
  - Lift index
- Grid-based and regional aggregation options.
- Supports both global and regional domain analysis.
- Data visualization.

The step2 and step3 is additional exploration, not being the main function.

This repository contains code developed during ongoing research.
The dataset used in the experiments is not publicly available.
Therefore, the results reported in the related manuscript cannot be 
fully reproduced with this code alone. 

This code is provided for demonstration and academic review purposes only.
