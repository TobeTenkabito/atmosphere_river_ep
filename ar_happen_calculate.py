import matplotlib
matplotlib.use('TkAgg')
import netCDF4 as nc
import numpy as np
from netCDF4 import num2date, date2num
from datetime import datetime, date

# Open dataset
ds = nc.Dataset('G:/Ar_dataverse/globalARcatalog_ERA5_1940-2024_v4.0.nc')

# Get time and spatial variables
time = ds.variables['time'][:]
time_units = ds.variables['time'].units
time_calendar = ds.variables['time'].calendar
dates = num2date(time, units=time_units, calendar=time_calendar)
lons_global = ds.variables['lon'][:]
lats_global = ds.variables['lat'][:]

print(f"Dataset time range: {dates[0]} to {dates[-1]}")
print(f"Total time steps: {len(time)}")
print(f"Longitude range: {lons_global[0]} to {lons_global[-1]}")
print(f"Latitude range: {lats_global[0]} to {lats_global[-1]}")

# User input for start and end dates
print("请输入起始日期（格式：YYYY-MM-DD HH:MM:SS，例如 2020-06-15 00:00:00）：")
start_date_input = input("起始日期：")
print("请输入结束日期（格式：YYYY-MM-DD HH:MM:SS，例如 2023-06-15 00:00:00）：")
end_date_input = input("结束日期：")

try:
    start_date = datetime.strptime(start_date_input, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(end_date_input, "%Y-%m-%d %H:%M:%S")
    start_time_num = date2num(start_date, units=time_units, calendar=time_calendar)
    end_time_num = date2num(end_date, units=time_units, calendar=time_calendar)

    if start_time_num < time[0] or end_time_num > time[-1]:
        raise ValueError(f"输入的日期超出数据范围（{dates[0]} to {dates[-1]}）！")
    if start_date > end_date:
        raise ValueError("起始日期不能晚于结束日期！")

    start_time_idx = np.argmin(np.abs(time - start_time_num))
    end_time_idx = np.argmin(np.abs(time - end_time_num))
    actual_start_date = dates[start_time_idx]
    actual_end_date = dates[end_time_idx]
    print(f"选择的起始日期：{actual_start_date} (Time Index {start_time_idx})")
    print(f"选择的结束日期：{actual_end_date} (Time Index {end_time_idx})")

except ValueError as e:
    print(f"错误：{e}")
    print("请使用正确的日期格式（YYYY-MM-DD HH:MM:SS）或确保日期在数据范围内。")
    ds.close()
    exit()

# 6-hour time step
step = 1
time_indices = np.arange(start_time_idx, end_time_idx + 1, step)

# Region and grid
lon_min, lon_max = 100, 150
lat_min, lat_max = 10, 60

# Find indices for the region in global grid
lon_idx = (lons_global >= lon_min) & (lons_global <= lon_max)
lat_idx = (lats_global >= lat_min) & (lats_global <= lat_max)
lons = lons_global[lon_idx]
lats = lats_global[lat_idx][::-1]  # South to north
n_lon, n_lat = len(lons), len(lats)

# Verify grid
print(f"Output grid: {n_lon}x{n_lat} points")
print(f"Longitude: {lons[0]} to {lons[-1]}")
print(f"Latitude: {lats[0]} to {lats[-1]}")
if not np.allclose(np.diff(lons), 0.25) or not np.allclose(np.diff(lats), 0.25):
    print("警告：网格分辨率不一致！")
    ds.close()
    exit()

# Initialize output array
n_time = len(time_indices)
ar_happen = np.zeros((n_time, n_lat, n_lon), dtype=np.int8)

# Process shapemap
zero_steps = 0
for t_idx, idx in enumerate(time_indices):
    shapemap = ds.variables['shapemap'][0, idx, 0, lat_idx, lon_idx][::-1, :]
    non_nan_count = np.sum(~np.isnan(shapemap))
    unique_vals = np.unique(shapemap[~np.isnan(shapemap)])
    print(f"Time {dates[idx]}: {len(unique_vals)} unique AR objects, {non_nan_count}/{n_lat*n_lon} grid points covered")
    ar_happen[t_idx, :, :] = (~np.isnan(shapemap)).astype(np.int8)
    if non_nan_count == 0:
        zero_steps += 1

# Create output NetCDF file
out_ds = nc.Dataset('G:/ar_analysis/ar_happen.nc', 'w', format='NETCDF4')
out_ds.createDimension('time', n_time)
out_ds.createDimension('lat', n_lat)
out_ds.createDimension('lon', n_lon)

time_var = out_ds.createVariable('time', 'f8', ('time',))
time_var.units = time_units
time_var.calendar = time_calendar
time_var[:] = time[time_indices]

lat_var = out_ds.createVariable('lat', 'f4', ('lat',))
lat_var.units = 'degrees_north'
lat_var[:] = lats

lon_var = out_ds.createVariable('lon', 'f4', ('lon',))
lon_var.units = 'degrees_east'
lon_var[:] = lons

ar_happen_var = out_ds.createVariable('ar_happen', 'i1', ('time', 'lat', 'lon'))
ar_happen_var.units = '1'
ar_happen_var.description = 'AR occurrence (1=occur, 0=not occur, based on shapemap)'
ar_happen_var[:] = ar_happen

out_ds.close()

# Statistics
total_occurrences = np.sum(ar_happen)
total_grid_cells = n_time * n_lat * n_lon
ar_grid_percent = (total_occurrences / total_grid_cells) * 100 if total_grid_cells > 0 else 0.0

# Calculate days and time steps with AR (threshold: >= 400 grid points)
unique_days = set()
ar_time_steps = 0
for t_idx, idx in enumerate(time_indices):
    ar_grid_count = np.sum(ar_happen[t_idx, :, :])  # Count AR grid points at this time step
    if ar_grid_count >= 400:  # Threshold for valid AR time step
        ar_time_steps += 1
        current_date = date(dates[idx].year, dates[idx].month, dates[idx].day)
        unique_days.add(current_date)

days_with_ar = len(unique_days)
ar_time_percent = (ar_time_steps / n_time) * 100 if n_time > 0 else 0.0

print(f"Total AR occurrences (grid-time): {total_occurrences}")
print(f"AR grid cells percentage: {ar_grid_percent:.2f}%")
print(f"Days with AR occurrence (≥400 grid points): {days_with_ar}")
print(f"AR time steps percentage (≥400 grid points): {ar_time_percent:.2f}%")
print(f"Zero AR steps: {zero_steps}/{n_time} ({zero_steps/n_time*100:.2f}%)")

# Check uniform data
for t_idx, idx in enumerate(time_indices):
    unique_vals = np.unique(ar_happen[t_idx, :, :])
    if len(unique_vals) == 1:
        print(f"警告：Time {dates[idx]} has uniform ar_happen values: {unique_vals}")

# Close dataset
ds.close()