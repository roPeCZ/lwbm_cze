# Lumped Water Balance Model - Czech Republic region
Lumped water balance model developed for the Central European regions regarding the climate change effects on hydrological conditions.
This is a repository for the paper "Developing a new lumped water balance rainfall-runoff model in a daily timestep for the Central European regions: A case study of the Czech Republic". The usage of this code should be properly cited: https://doi.org/10.1016/j.envsoft.2024.106092. The repository was created by Martin Bednar (main author and corresponding author). If you have any questions about this model or code, you can contanct me via email Martin.Bednar2@vut.cz


# Sample data format
- Data are provided via MS Excel spreadsheet
- Calibration and validation data are within the same XSLX file but within separate sheets ("Calibration" and "Validation")
- The LWBM needs to be provided with mean daily temperature [°C], precipitation [mm], potential evapotranspiration [mm] and observed streamflow [mm]

# How to run a model calibration
- Edit 'Sample data.xlsx' file with your observed data (renaming columns will lead to code edit)
- Edit model parameters bounds values in the file 'parameters.json' if necessary
- In 'run.py' change argument for differential evolution algorithm (if necessary, probably increase number of generations)
- Run the file 'run.py' 
- After calibration end, validation dataset is evaluated
- Finally, calibration and validation results are saved in 'results.xlsx' file and optimised model parameters are saved in 'parameters.csv'
