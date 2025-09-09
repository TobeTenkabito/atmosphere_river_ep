import matplotlib

# Use the Agg backend for non-interactive plotting, which is more
# suitable for script-based image generation.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from netCDF4 import num2date, date2num
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import fisher_exact
from scipy.ndimage import uniform_filter
from tqdm import tqdm
import multiprocessing as mp
import os

# --- Configuration ---
spatial_smoothing_window_size = 1
sig_level = 0.05
lift_cap_percentile = 95
lift_max_val = 100
output_dir = "G:/ar_analysis/output_0"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


# Worker function for the multiprocessing pool.
def fisher_worker(args):
    """Worker function for the multiprocessing pool."""
    i, j, a, b, c, d = args
    table = [[a, b], [c, d]]
    if np.sum(table) > 0:
        _, p = fisher_exact(table)
        return i, j, p
    return i, j, np.nan


# The following code block ensures that this part of the script
# only runs in the main process, preventing duplicate output
# when child processes are spawned by `multiprocessing`.
if __name__ == "__main__":
    # --- Data Loading and Time Matching ---
    print("Reading NetCDF files...")
    try:
        ar_ds = nc.Dataset('G:/ar_analysis/ar_happen.nc', 'r')
        precip_ds = nc.Dataset('G:/ar_analysis/data/extreme_precipitation.nc', 'r')
    except FileNotFoundError as e:
        print(f"Error: One of the input files was not found. Please check the paths. {e}")
        exit()

    ar_time = ar_ds.variables['time'][:]
    ar_time_units = ar_ds.variables['time'].units
    ar_time_calendar = ar_ds.variables['time'].calendar
    ar_dates = num2date(ar_time, units=ar_time_units, calendar=ar_time_calendar)

    precip_time = precip_ds.variables['time'][:]
    precip_time_units = precip_ds.variables['time'].units
    precip_time_calendar = precip_ds.variables['time'].calendar
    precip_dates = num2date(precip_time, units=precip_time_units, calendar=precip_time_calendar)

    print(f"AR time range: {ar_dates[0]} to {ar_dates[-1]}")
    print(f"Precip time range: {precip_dates[0]} to {precip_dates[-1]}")

    # --- User Input and Time Indexing ---
    try:
        start_date_input = input("起始日期（YYYY-MM-DD HH:MM:SS）: ")
        end_date_input = input("结束日期（YYYY-MM-DD HH:MM:SS）: ")
        start_date = datetime.strptime(start_date_input, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(end_date_input, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print("日期格式不正确，请使用 YYYY-MM-DD HH:MM:SS 格式。")
        exit()

    print("Matching dates and finding indices (Optimized)...")
    # Find the indices for the specified time range using searchsorted
    ar_idx_sel = np.searchsorted(ar_dates, start_date), np.searchsorted(ar_dates, end_date)
    precip_idx_sel = np.searchsorted(precip_dates, start_date), np.searchsorted(precip_dates, end_date)

    # Slice dates to find common times and ensure they are aligned
    ar_dates_sel = ar_dates[ar_idx_sel[0]:ar_idx_sel[1]]
    precip_dates_sel = precip_dates[precip_idx_sel[0]:precip_idx_sel[1]]

    # Create a dictionary for fast lookup of precip dates
    precip_date_to_idx = {date: idx for idx, date in enumerate(precip_dates)}
    ar_indices = []
    precip_indices = []
    common_dates = []

    # Use tqdm to show progress for the date matching loop
    for ar_date in tqdm(ar_dates_sel, desc="Matching dates"):
        if ar_date in precip_date_to_idx:
            ar_indices.append(np.where(ar_dates == ar_date)[0][0])
            precip_indices.append(precip_date_to_idx[ar_date])
            common_dates.append(ar_date)

    ar_indices = np.array(ar_indices)
    precip_indices = np.array(precip_indices)
    common_dates = np.array(common_dates)

    if len(common_dates) == 0:
        print("没有共同时间！")
        exit()

    n_time = len(common_dates)
    print(f"共同时间步长：{n_time}")

    # --- Spatial Indexing ---
    lon_min, lon_max = 100, 150
    lat_min, lat_max = 10, 60
    grid_res = 0.25

    precip_lats_all = precip_ds.variables['latitude'][:]
    precip_lons_all = precip_ds.variables['longitude'][:]
    ar_lats_all = ar_ds.variables['lat'][:]
    ar_lons_all = ar_ds.variables['lon'][:]

    # Find indices for the geographical bounding box
    lat_indices = np.where((precip_lats_all >= lat_min) & (precip_lats_all <= lat_max))[0]
    lon_indices = np.where((precip_lons_all >= lon_min) & (precip_lons_all <= lon_max))[0]
    ar_lat_indices = np.where((ar_lats_all >= lat_min) & (ar_lats_all <= lat_max))[0]
    ar_lon_indices = np.where((ar_lons_all >= lon_min) & (ar_lons_all <= lon_max))[0]

    precip_lats = precip_lats_all[lat_indices]
    precip_lons = precip_lons_all[lon_indices]
    ar_lats = ar_lats_all[ar_lat_indices]
    ar_lons = ar_lons_all[ar_lon_indices]

    # Handle potential latitude inversions
    if ar_lats[0] > ar_lats[-1]:
        ar_lats = ar_lats[::-1]
        ar_lat_indices = ar_lat_indices[::-1]
    if precip_lats[0] > precip_lats[-1]:
        precip_lats = precip_lats[::-1]
        lat_indices = lat_indices[::-1]

    # --- Optimized Data Reading and Calculation ---
    # Read only the necessary slices to conserve memory.
    print("Reading data slices...")
    ar_data = ar_ds.variables['ar_happen'][ar_indices, ar_lat_indices, ar_lon_indices].astype(np.int8)
    precip_data = precip_ds.variables['extreme_precipitation_flag'][precip_indices, lat_indices, lon_indices].astype(
        np.int8)

    print("Calculating contingency table...")
    A = np.sum((ar_data == 1) & (precip_data == 1), axis=0)
    B = np.sum((ar_data == 1) & (precip_data == 0), axis=0)
    C = np.sum((ar_data == 0) & (precip_data == 1), axis=0)
    D = np.sum((ar_data == 0) & (precip_data == 0), axis=0)

    # Clean up memory from large intermediate arrays
    del ar_data, precip_data

    p_ext_ar = np.where((A + B) > 0, A / (A + B), np.nan)
    p_ext_no_ar = np.where((C + D) > 0, C / (C + D), np.nan)
    lift = np.where((p_ext_no_ar > 0), p_ext_ar / p_ext_no_ar, np.nan)
    lift = np.where(lift > lift_max_val, lift_max_val, lift)

    # --- Parallelized Fisher Exact Test with Progress Bar ---
    print("Running Fisher Exact Test (Parallelized)...")
    p_sig = np.full(A.shape, np.nan)
    coords = [(i, j, A[i, j], B[i, j], C[i, j], D[i, j])
              for i in range(A.shape[0]) for j in range(A.shape[1])]

    # Use a multiprocessing pool to parallelize the slow Fisher exact test loop.
    # tqdm is used to wrap the pool.imap() call, providing a real-time progress bar.
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(fisher_worker, coords), total=len(coords)))

    # Unpack results into the p_sig grid
    for i, j, p in results:
        p_sig[i, j] = p

    # --- Spatial Smoothing (Sliding Window Average) ---
    if spatial_smoothing_window_size > 1:
        print(f"Applying spatial smoothing with window size {spatial_smoothing_window_size}...")
        p_ext_ar = uniform_filter(p_ext_ar, size=spatial_smoothing_window_size, mode='nearest')
        p_ext_no_ar = uniform_filter(p_ext_no_ar, size=spatial_smoothing_window_size, mode='nearest')
        lift = uniform_filter(lift, size=spatial_smoothing_window_size, mode='nearest')
        p_sig = uniform_filter(p_sig, size=spatial_smoothing_window_size, mode='nearest')

    prop_ar_extreme = np.where((A + C) > 0, A / (A + C), np.nan)
    mean_prop_ar_extreme = np.nanmean(prop_ar_extreme)


    # --- Plotting Function ---
    def plot_map(data, title, filename, vmin=None, vmax=None, sig_mask=None):
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#D3D3D3')
        ax.add_feature(cfeature.OCEAN, facecolor='#E6F0FA')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
        ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)

        if vmin is None or vmax is None:
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                vmin = np.nanpercentile(valid_data, 0) if vmin is None else vmin
                vmax = np.nanpercentile(valid_data, 100) if vmax is None else vmax
            else:
                vmin, vmax = 0, 1

        n_levels = 7
        levels = np.linspace(vmin, vmax, n_levels + 1)
        cmap = plt.get_cmap('viridis', n_levels)

        # Apply significance mask for plotting
        masked_data = np.ma.masked_where(sig_mask >= sig_level, data) if sig_mask is not None else data
        mesh = ax.contourf(precip_lons, precip_lats, masked_data, levels=levels, cmap=cmap,
                           transform=ccrs.PlateCarree())

        # Add scatter plot for non-significant areas
        if sig_mask is not None:
            nonsig_lats, nonsig_lons = np.where(sig_mask >= sig_level)
            ax.scatter(precip_lons[nonsig_lons], precip_lats[nonsig_lats], s=10, color='gray', marker='.', alpha=0.5,
                       transform=ccrs.PlateCarree())

        cbar = plt.colorbar(mesh, ax=ax, label=title, shrink=0.8, ticks=levels)
        cbar.set_label(title, fontsize=10)
        plt.title(f"{title}\n{start_date_input} to {end_date_input}", fontsize=14)
        plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
        plt.close()


    # --- Generate Plots and Summary Statistics ---
    print("Generating plots...")
    # Now the plots for conditional probabilities will also be auto-adjusted
    plot_map(p_ext_ar, 'P(Extreme | AR)', 'p_ext_ar', sig_mask=p_sig)
    plot_map(p_ext_no_ar, 'P(Extreme | No AR)', 'p_ext_no_ar', sig_mask=p_sig)
    plot_map(lift, 'Lift [P(Extreme | AR) / P(Extreme | No AR)]', 'lift', sig_mask=p_sig)

    with open(f"{output_dir}/summary_stats.txt", "w") as f:
        f.write(f"Mean P(Extreme | AR): {np.nanmean(p_ext_ar):.3f}\n")
        f.write(f"Mean P(Extreme | No AR): {np.nanmean(p_ext_no_ar):.3f}\n")
        f.write(f"Mean Lift: {np.nanmean(lift):.3f}\n")
        f.write(f"Proportion of Extreme Precipitation Events with AR: {mean_prop_ar_extreme:.3f}\n")
        f.write(f"Effective grid points: {np.sum(~np.isnan(lift))}\n")
        f.write(f"Significant grid points (p < {sig_level}): {np.sum(p_sig < sig_level)}\n")

    print("绘图与统计完成，输出文件已保存到:", output_dir)

    ar_ds.close()
    precip_ds.close()
