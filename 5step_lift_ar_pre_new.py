import matplotlib

# Use the Agg backend for non-interactive plotting, which is more
# suitable for script-based image generation.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from netCDF4 import num2date
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import fisher_exact
from scipy.ndimage import uniform_filter
from tqdm import tqdm
import multiprocessing as mp
from matplotlib.colors import LinearSegmentedColormap
import os
import itertools  # Import itertools for sequential Fisher test
import psutil

# --- Configuration ---
spatial_smoothing_window_size = 1
sig_level = 0.05
lift_cap_percentile = 95
lift_max_val = 100
bootstrap_iterations = 1000  # Number of bootstrap samples for CI calculation
base_output_dir = "G:/ar_analysis/output_seasonal_regional_1"

# --- File Paths (for passing to parallel workers) ---
AR_FILE_PATH = '/ar_happen_ERA5.nc'
PRECIP_FILE_PATH = 'G:/ar_analysis/data/extreme_precipitation.nc'

# Create base output directory if it doesn't exist
os.makedirs(base_output_dir, exist_ok=True)

# --- Region Definitions ---
regions = {
    "All_East_Asia": {"lon_min": 100, "lon_max": 150, "lat_min": 10, "lat_max": 60, "name": "East_Asia"},
    "South_China": {"lon_min": 105, "lon_max": 125, "lat_min": 18, "lat_max": 35, "name": "South_China"},
    "North_China": {"lon_min": 105, "lon_max": 125, "lat_min": 32, "lat_max": 45, "name": "North_China"},
    "Korea_Japan_Archipelago": {"lon_min": 122, "lon_max": 146, "lat_min": 30, "lat_max": 50, "name": "Korea_Japan"},
    "Northeast_Asia": {"lon_min": 120, "lon_max": 140, "lat_min": 40, "lat_max": 60, "name": "Northeast_Asia"}
}

# --- Season Definitions ---
seasons = {
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "DJF": [12, 1, 2]
}


def bootstrap_worker(n_iter, ar_data, precip_data, n_timesteps, D):
    """单个进程执行 n_iter 次 bootstrap"""
    paf_list, af_list = [], []
    rng = np.random.default_rng()  # 每个 worker 独立 RNG

    for _ in range(n_iter):
        resample_indices = rng.integers(0, n_timesteps, size=n_timesteps)

        A_boot = np.sum((ar_data[resample_indices] == 1) & (precip_data[resample_indices] == 1), axis=0)
        B_boot = np.sum((ar_data[resample_indices] == 1) & (precip_data[resample_indices] == 0), axis=0)
        C_boot = np.sum((ar_data[resample_indices] == 0) & (precip_data[resample_indices] == 1), axis=0)

        p_ext_ar_boot = np.where((A_boot + B_boot) > 0, A_boot / (A_boot + B_boot), np.nan)
        p_ext_no_ar_boot = np.where((C_boot + D) > 0, C_boot / (C_boot + D), np.nan)

        frequency_e_boot = (A_boot + C_boot) / n_timesteps
        p_no_ar_boot = (C_boot + D) / n_timesteps

        paf_boot = np.where(frequency_e_boot > 0,
                            (frequency_e_boot - p_no_ar_boot * p_ext_no_ar_boot) / frequency_e_boot, np.nan)
        af_boot = np.where(p_ext_ar_boot > 0, (p_ext_ar_boot - p_ext_no_ar_boot) / p_ext_ar_boot, np.nan)

        paf_list.append(np.nanmean(paf_boot))
        af_list.append(np.nanmean(af_boot))

    return paf_list, af_list


def adaptive_num_workers(memory_per_worker_gb=2):
    """根据系统可用内存动态分配进程数"""
    free_mem = psutil.virtual_memory().available / (1024 ** 3)
    max_workers = int(free_mem // memory_per_worker_gb)
    return max(1, min(max_workers, mp.cpu_count()))


def run_bootstrap(ar_data, precip_data, n_timesteps, D, bootstrap_iterations, analysis_name):
    print(f"Running bootstrap ({bootstrap_iterations} iters) for {analysis_name} (single-core)...")

    paf_list, af_list = [], []
    rng = np.random.default_rng()

    for _ in tqdm(range(bootstrap_iterations), desc=f"Bootstrap for {analysis_name}"):
        resample_indices = rng.integers(0, n_timesteps, size=n_timesteps)

        A_boot = np.sum((ar_data[resample_indices] == 1) & (precip_data[resample_indices] == 1), axis=0)
        B_boot = np.sum((ar_data[resample_indices] == 1) & (precip_data[resample_indices] == 0), axis=0)
        C_boot = np.sum((ar_data[resample_indices] == 0) & (precip_data[resample_indices] == 1), axis=0)

        p_ext_ar_boot = np.where((A_boot + B_boot) > 0, A_boot / (A_boot + B_boot), np.nan)
        p_ext_no_ar_boot = np.where((C_boot + D) > 0, C_boot / (C_boot + D), np.nan)

        frequency_e_boot = (A_boot + C_boot) / n_timesteps
        p_no_ar_boot = (C_boot + D) / n_timesteps

        paf_boot = np.where(frequency_e_boot > 0,
                            (frequency_e_boot - p_no_ar_boot * p_ext_no_ar_boot) / frequency_e_boot, np.nan)
        af_boot = np.where(p_ext_ar_boot > 0, (p_ext_ar_boot - p_ext_no_ar_boot) / p_ext_ar_boot, np.nan)

        paf_list.append(np.nanmean(paf_boot))
        af_list.append(np.nanmean(af_boot))

    paf_ci_95 = np.percentile(paf_list, [2.5, 97.5])
    af_ci_95 = np.percentile(af_list, [2.5, 97.5])

    return paf_ci_95, af_ci_95


def plot_map(data, title, filename, output_dir, region_name, vmin=None, vmax=None, sig_mask=None, cmap='viridis',
             n_levels=7, lon_min=100, lon_max=150, lat_min=10, lat_max=60, lons=None, lats=None, date_range_str=""):
    """Enhanced plotting function to handle regional plotting."""
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
            vmin_calc = np.nanmin(valid_data) if vmin is None else vmin
            vmax_calc = np.nanmax(valid_data) if vmax is None else vmax
            # Apply the user's requested lower bound for PAF and AF
            if title.startswith("PAF") or title.startswith("AF"):
                vmin_calc = max(0, vmin_calc)
            vmin = vmin_calc
            vmax = vmax_calc
        else:
            vmin, vmax = 0, 1

    levels = np.linspace(vmin, vmax, n_levels + 1)

    mesh = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree())

    ax.contour(lons, lats, data, levels=levels, colors='black', linewidths=0.5, transform=ccrs.PlateCarree())

    if sig_mask is not None:
        significant_mask = np.where(sig_mask < sig_level, 1, np.nan)
        ax.contourf(lons, lats, significant_mask, levels=[0.5, 1.5],
                    colors='none', hatches=['...'], transform=ccrs.PlateCarree())

    cbar = plt.colorbar(mesh, ax=ax, label=title, shrink=0.8)
    cbar.set_label(title, fontsize=10)

    full_title = f"{title} for {region_name}\n{date_range_str}"
    plt.title(full_title, fontsize=14)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_seasonal_subfigures(seasonal_data, output_dir, filename, title, lon_min, lon_max, lat_min, lat_max, lons,
                             lats):
    """
    Plots a 2x2 grid of subfigures for seasonal analysis of a single metric.
    """
    all_data = np.concatenate([d.flatten() for d in seasonal_data.values()])
    vmin = np.nanmin(all_data) if not np.all(np.isnan(all_data)) else 0
    vmax = np.nanmax(all_data) if not np.all(np.isnan(all_data)) else 1

    if title in ["PAF", "AF"]:
        vmin = 0

    cmap = 'viridis' if title == "Lift" else 'coolwarm'

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    seasons_list = list(seasons.keys())

    for i, season_name in enumerate(seasons_list):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        data = seasonal_data.get(season_name, np.full(lats.shape, np.nan))  # Handle missing seasons

        mesh = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#D3D3D3')
        ax.add_feature(cfeature.OCEAN, facecolor='#E6F0FA')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)

        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (col == 0)
        gl.bottom_labels = (row == 1)

        ax.text(0.02, 0.98, subplot_labels[i], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')
        ax.text(0.98, 0.98, season_name, transform=ax.transAxes,
                fontsize=12, va='top', ha='right')

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'{title} Value', fontsize=12)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_seasonal_subfigures(seasonal_results, output_dir, lon_min, lon_max, lat_min, lat_max, lons, lats):
    """
    Orchestrates plotting for all metrics based on collected seasonal results.
    """
    metrics_to_plot = {
        "p_ext_ar": "P(Extreme | AR)",
        "p_ext_no_ar": "P(Extreme | No AR)",
        "paf": "PAF",
        "af": "AF",
        "lift": "Lift"
    }

    # Restructure data from a list of dicts to a dict of dicts
    seasonal_data_dict = {metric: {} for metric in metrics_to_plot.keys()}
    for result in seasonal_results:
        if result:  # Check if result is not None
            season_name = result['analysis_name'].split('_')[-1]
            for metric_key in metrics_to_plot.keys():
                seasonal_data_dict[metric_key][season_name] = result['metrics'][metric_key]

    for metric_key, metric_title in metrics_to_plot.items():
        if not seasonal_data_dict[metric_key]:
            print(f"Skipping combined plot for {metric_title} due to missing seasonal data.")
            continue

        filename = f"combined_seasonal_{metric_key}"
        combined_output_dir = os.path.join(output_dir, "seasonal_combined")

        print(f"Generating combined seasonal subplot for: {metric_title}")
        plot_seasonal_subfigures(seasonal_data_dict[metric_key], combined_output_dir, filename, metric_title,
                                 lon_min, lon_max, lat_min, lat_max, lons, lats)


def run_analysis(ar_indices, precip_indices,
                 ar_lat_indices, ar_lon_indices,
                 precip_lat_indices, precip_lon_indices,
                 precip_lats, precip_lons,
                 ar_ds, precip_ds,
                 output_dir, analysis_name,
                 lon_min, lon_max, lat_min, lat_max,
                 date_range_str):
    print(f"\n--- Starting Analysis for: {analysis_name} ---")

    if len(ar_indices) == 0:
        print(f"Skipping {analysis_name} due to no common time steps.")
        return None

    # --- Data ---
    print("Reading data slices...")
    ar_data = ar_ds.variables['ar_happen'][ar_indices, ar_lat_indices, ar_lon_indices].astype(np.int8)
    precip_data = precip_ds.variables['extreme_precipitation_flag'][
        precip_indices, precip_lat_indices, precip_lon_indices].astype(np.int8)

    print("Calculating contingency table...")
    A = np.sum((ar_data == 1) & (precip_data == 1), axis=0)
    B = np.sum((ar_data == 1) & (precip_data == 0), axis=0)
    C = np.sum((ar_data == 0) & (precip_data == 1), axis=0)
    D = np.sum((ar_data == 0) & (precip_data == 0), axis=0)

    # --- Bootstrap optimization ---
    n_timesteps = len(ar_indices)
    paf_ci_95, af_ci_95 = run_bootstrap(ar_data, precip_data, n_timesteps, D, bootstrap_iterations, analysis_name)

    del ar_data, precip_data

    # --- Calculate Main Metrics ---
    p_no_ar = np.where((A + B + C + D) > 0, (C + D) / (A + B + C + D), np.nan)
    frequency_e = np.where((A + B + C + D) > 0, (A + C) / (A + B + C + D), np.nan)
    p_ext_ar = np.where((A + B) > 0, A / (A + B), np.nan)
    p_ext_no_ar = np.where((C + D) > 0, C / (C + D), np.nan)
    paf = np.where(frequency_e > 0, (frequency_e - p_no_ar * p_ext_no_ar) / frequency_e, np.nan)
    af = np.where(p_ext_ar > 0, (p_ext_ar - p_ext_no_ar) / p_ext_ar, np.nan)
    lift = np.where((p_ext_no_ar > 0), p_ext_ar / frequency_e, np.nan)
    #lift_upper_bound = np.nanpercentile(lift, lift_cap_percentile)
    #lift = np.where(lift > lift_upper_bound, lift_upper_bound, lift)
    paf = np.where(paf < 0, np.nan, paf)
    af = np.where(af < 0, np.nan, af)
    lift = np.where(lift < 0, np.nan, lift)


    # --- Fisher Exact Test (Sequential) ---
    p_sig = np.full(A.shape, np.nan)
    for i, j in itertools.product(range(A.shape[0]), range(A.shape[1])):
        table = [[A[i, j], B[i, j]], [C[i, j], D[i, j]]]
        if np.sum(table) > 0:
            _, p = fisher_exact(table)
            p_sig[i, j] = p

    # --- Spatial Smoothing ---
    if spatial_smoothing_window_size > 1:
        p_ext_ar = uniform_filter(p_ext_ar, size=spatial_smoothing_window_size, mode='nearest')
        p_ext_no_ar = uniform_filter(p_ext_no_ar, size=spatial_smoothing_window_size, mode='nearest')
        paf = uniform_filter(paf, size=spatial_smoothing_window_size, mode='nearest')
        af = uniform_filter(af, size=spatial_smoothing_window_size, mode='nearest')
        lift = uniform_filter(lift, size=spatial_smoothing_window_size, mode='nearest')
        p_sig = uniform_filter(p_sig, size=spatial_smoothing_window_size, mode='nearest')

    sig_mask_bool = p_sig < sig_level
    metrics = {
        "p_ext_ar": np.where(sig_mask_bool, p_ext_ar, np.nan),
        "p_ext_no_ar": np.where(sig_mask_bool, p_ext_no_ar, np.nan),
        "paf": np.where(sig_mask_bool, paf, np.nan),
        "af": np.where(sig_mask_bool, af, np.nan),
        "lift": np.where(sig_mask_bool, lift, np.nan)
    }

    # --- Generate Plots ---
    plot_map(metrics["p_ext_ar"], 'P(Extreme | AR) (Significant Areas)', 'p_ext_ar_sig', output_dir, analysis_name,
             lons=precip_lons, lats=precip_lats, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
             date_range_str=date_range_str)
    plot_map(metrics["p_ext_no_ar"], 'P(Extreme | No AR) (Significant Areas)', 'p_ext_no_ar_sig', output_dir,
             analysis_name, lons=precip_lons, lats=precip_lats, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min,
             lat_max=lat_max, date_range_str=date_range_str)
    plot_map(metrics["paf"], "PAF (Significant Areas)", "paf_sig", output_dir, analysis_name, cmap="coolwarm",
             lons=precip_lons, lats=precip_lats, lon_min=lon_min, lon_max=lon_max, vmin=0, lat_min=lat_min,
             lat_max=lat_max, date_range_str=date_range_str)
    plot_map(metrics["af"], "AF (Significant Areas)", "af_sig", output_dir, analysis_name, vmin=0, cmap="coolwarm",
             lons=precip_lons, lats=precip_lats, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
             date_range_str=date_range_str)
    plot_map(metrics["lift"], 'Lift (Significant Areas)', 'lift_sig', output_dir, analysis_name, lons=precip_lons,
             lats=precip_lats, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
             date_range_str=date_range_str)

    # --- Summary Statistics ---
    prop_ar_extreme = np.where((A + C) > 0, A / (A + C), np.nan)
    mean_prop_ar_extreme = np.nanmean(prop_ar_extreme)

    with open(f"{output_dir}/summary_stats_{analysis_name}.txt", "w") as f:
        f.write(f"--- Summary Statistics for {analysis_name} ---\n")
        f.write(f"Date Range: {date_range_str}\n\n")
        for name, data in metrics.items():
            f.write(f"--- {name} ---\n")
            f.write(f"  Mean: {np.nanmean(data):.4f}\n")
            if name == 'paf': f.write(f"  Mean PAF 95% CI: [{paf_ci_95[0]:.4f}, {paf_ci_95[1]:.4f}]\n")
            if name == 'af': f.write(f"  Mean AF 95% CI: [{af_ci_95[0]:.4f}, {af_ci_95[1]:.4f}]\n")
            f.write(f"  Min:  {np.nanmin(data):.4f}\n")
            f.write(f"  Max:  {np.nanmax(data):.4f}\n\n")
        f.write("--- Other Stats ---\n")
        f.write(f"Proportion of Extreme Precipitation Events with AR: {mean_prop_ar_extreme:.4f}\n")
        f.write(f"Effective grid points: {np.sum(~np.isnan(lift))}\n")
        f.write(f"Significant grid points (p < {sig_level}): {np.sum(p_sig < sig_level)}\n")

    print(f"--- [PID:{os.getpid()}] Analysis for {analysis_name} complete. ---")

    # Return metrics for final plotting
    return {"analysis_name": analysis_name, "metrics": metrics}


def main_worker(task_params):
    """
    Worker function to unpack parameters and run analysis.
    Opens datasets within the worker process.
    """
    ar_ds = nc.Dataset(AR_FILE_PATH, 'r')
    precip_ds = nc.Dataset(PRECIP_FILE_PATH, 'r')

    result = run_analysis(**task_params, ar_ds=ar_ds, precip_ds=precip_ds)

    ar_ds.close()
    precip_ds.close()
    return result


if __name__ == "__main__":
    try:
        start_date_input = input("here input start date（YYYY-MM-DD HH:MM:SS）: ")
        end_date_input = input("here input end date（YYYY-MM-DD HH:MM:SS）: ")
        start_date = datetime.strptime(start_date_input, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(end_date_input, "%Y-%m-%d %H:%M:%S")
        date_range_str_for_output = f"{start_date_input} to {end_date_input}"
    except ValueError:
        print("wrong with data format，please follow YYYY-MM-DD HH:MM:SS")
        exit()

    # --- Data Loading and Time Matching (in main process) ---
    print("Reading and matching time coordinates...")
    with nc.Dataset(AR_FILE_PATH, 'r') as ar_ds, nc.Dataset(PRECIP_FILE_PATH, 'r') as precip_ds:
        ar_time = ar_ds.variables['time'][:]
        ar_dates = num2date(ar_time, units=ar_ds.variables['time'].units, calendar=ar_ds.variables['time'].calendar)
        precip_time = precip_ds.variables['time'][:]
        precip_dates = num2date(precip_time, units=precip_ds.variables['time'].units,
                                calendar=precip_ds.variables['time'].calendar)

        precip_lats_all = precip_ds.variables['latitude'][:]
        precip_lons_all = precip_ds.variables['longitude'][:]
        ar_lats_all = ar_ds.variables['lat'][:]
        ar_lons_all = ar_ds.variables['lon'][:]

    precip_date_to_idx = {date: idx for idx, date in enumerate(precip_dates)}
    all_ar_indices, all_precip_indices, common_dates = [], [], []
    for idx, ar_date in enumerate(ar_dates):
        if start_date <= ar_date <= end_date and ar_date in precip_date_to_idx:
            all_ar_indices.append(idx)
            all_precip_indices.append(precip_date_to_idx[ar_date])
            common_dates.append(ar_date)

    if not common_dates:
        print("no time have something in common")
        exit()

    print(f"common time step：{len(common_dates)}")

    # --- Prepare all analysis tasks ---
    tasks = []

    # 1. Regional analysis tasks
    for region_key, region_props in regions.items():
        lon_min, lon_max, lat_min, lat_max = region_props["lon_min"], region_props["lon_max"], region_props["lat_min"], \
        region_props["lat_max"]

        precip_lat_indices = np.where((precip_lats_all >= lat_min) & (precip_lats_all <= lat_max))[0]
        precip_lon_indices = np.where((precip_lons_all >= lon_min) & (precip_lons_all <= lon_max))[0]
        ar_lat_indices = np.where((ar_lats_all >= lat_min) & (ar_lats_all <= lat_max))[0]
        ar_lon_indices = np.where((ar_lons_all >= lon_min) & (ar_lons_all <= lon_max))[0]

        precip_lats_region = precip_lats_all[precip_lat_indices]
        if precip_lats_region.size > 1 and precip_lats_region[0] > precip_lats_region[-1]:
            precip_lat_indices = precip_lat_indices[::-1]

        tasks.append({
            "ar_indices": all_ar_indices, "precip_indices": all_precip_indices,
            "ar_lat_indices": ar_lat_indices, "ar_lon_indices": ar_lon_indices,
            "precip_lat_indices": precip_lat_indices, "precip_lon_indices": precip_lon_indices,
            "precip_lats": precip_lats_all[precip_lat_indices], "precip_lons": precip_lons_all[precip_lon_indices],
            "output_dir": os.path.join(base_output_dir, region_props['name']),
            "analysis_name": region_props['name'], "lon_min": lon_min, "lon_max": lon_max,
            "lat_min": lat_min, "lat_max": lat_max, "date_range_str": date_range_str_for_output
        })

    # 2. Seasonal analysis tasks (Global Scope)
    global_region = regions["All_East_Asia"]
    lon_min_g, lon_max_g, lat_min_g, lat_max_g = global_region["lon_min"], global_region["lon_max"], global_region[
        "lat_min"], global_region["lat_max"]
    precip_lat_indices_g = np.where((precip_lats_all >= lat_min_g) & (precip_lats_all <= lat_max_g))[0]
    precip_lon_indices_g = np.where((precip_lons_all >= lon_min_g) & (precip_lons_all <= lon_max_g))[0]
    ar_lat_indices_g = np.where((ar_lats_all >= lat_min_g) & (ar_lats_all <= lat_max_g))[0]
    ar_lon_indices_g = np.where((ar_lons_all >= lon_min_g) & (ar_lons_all <= lon_max_g))[0]

    precip_lats_global = precip_lats_all[precip_lat_indices_g]
    if precip_lats_global.size > 1 and precip_lats_global[0] > precip_lats_global[-1]:
        precip_lat_indices_g = precip_lat_indices_g[::-1]

    for season_name, months in seasons.items():
        seasonal_mask = np.array([date.month in months for date in common_dates])
        tasks.append({
            "ar_indices": np.array(all_ar_indices)[seasonal_mask],
            "precip_indices": np.array(all_precip_indices)[seasonal_mask],
            "ar_lat_indices": ar_lat_indices_g, "ar_lon_indices": ar_lon_indices_g,
            "precip_lat_indices": precip_lat_indices_g, "precip_lon_indices": precip_lon_indices_g,
            "precip_lats": precip_lats_all[precip_lat_indices_g], "precip_lons": precip_lons_all[precip_lon_indices_g],
            "output_dir": os.path.join(base_output_dir, "seasonal", season_name),
            "analysis_name": f"Global_{season_name}", "lon_min": lon_min_g, "lon_max": lon_max_g,
            "lat_min": lat_min_g, "lat_max": lat_max_g, "date_range_str": date_range_str_for_output
        })

    # --- Run all tasks in parallel ---
    print(f"\nStarting parallel processing for {len(tasks)} tasks...")
    # Use context manager to ensure the pool is closed properly
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use tqdm to show a progress bar for the parallel tasks
        results = list(tqdm(pool.imap(main_worker, tasks), total=len(tasks)))

    # --- Post-processing: Generate combined seasonal plots ---
    print("\n--- Generating Combined Seasonal Subplots ---")
    seasonal_results = [res for res in results if res and 'Global' in res['analysis_name']]

    plot_all_seasonal_subfigures(seasonal_results, base_output_dir,
                                 lon_min_g, lon_max_g, lat_min_g, lat_max_g,
                                 precip_lons_all[precip_lon_indices_g], precip_lats_all[precip_lat_indices_g])

    print("\nAll analyses finished.")
