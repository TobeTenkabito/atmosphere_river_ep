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

# --- Configuration ---
spatial_smoothing_window_size = 1
sig_level = 0.05
lift_cap_percentile = 95
lift_max_val = 100
base_output_dir = "G:/ar_analysis/output_seasonal_regional"

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


# Worker function for the multiprocessing pool.
def fisher_worker(args):
    """Worker function for the multiprocessing pool."""
    i, j, a, b, c, d = args
    table = [[a, b], [c, d]]
    if np.sum(table) > 0:
        _, p = fisher_exact(table)
        return i, j, p
    return i, j, np.nan


def create_custom_cmap(colors, name='custom_cmap'):
    """Creates a custom colormap from a list of colors."""
    return LinearSegmentedColormap.from_list(name, colors)


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
            vmin = np.nanmin(valid_data) if vmin is None else vmin
            vmax = np.nanmax(valid_data) if vmax is None else vmax
        else:
            vmin, vmax = 0, 1

    levels = np.linspace(vmin, vmax, n_levels + 1)

    # Use pcolormesh for better representation of grid cell data
    mesh = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree())

    # Add contour lines for better visualization (optional but recommended)
    ax.contour(lons, lats, data, levels=levels, colors='black', linewidths=0.5, transform=ccrs.PlateCarree())

    if sig_mask is not None:
        # Use hatching for significant areas, ensuring it aligns with the pcolormesh grid
        significant_mask = np.where(sig_mask < sig_level, 1, np.nan)
        ax.contourf(lons, lats, significant_mask, levels=[0.5, 1.5],
                    colors='none', hatches=['...'], transform=ccrs.PlateCarree())

    cbar = plt.colorbar(mesh, ax=ax, label=title, shrink=0.8)
    cbar.set_label(title, fontsize=10)

    full_title = f"{title} for {region_name}\n{date_range_str}"
    plt.title(full_title, fontsize=14)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


def run_analysis(ar_indices, precip_indices, ar_lat_indices, ar_lon_indices, precip_lat_indices, precip_lon_indices,
                 precip_lats, precip_lons, ar_ds, precip_ds,
                 output_dir, analysis_name, lon_min, lon_max, lat_min, lat_max, date_range_str):
    """
    Encapsulates the core data processing, statistical analysis, and plotting logic.
    """
    print(f"\n--- Starting Analysis for: {analysis_name} ---")

    if len(ar_indices) == 0:
        print(f"Skipping {analysis_name} due to no common time steps.")
        return

    # --- Optimized Data Reading and Calculation ---
    print("Reading data slices...")
    ar_data = ar_ds.variables['ar_happen'][ar_indices, ar_lat_indices, ar_lon_indices].astype(np.int8)
    precip_data = precip_ds.variables['extreme_precipitation_flag'][
        precip_indices, precip_lat_indices, precip_lon_indices].astype(np.int8)

    print("Calculating contingency table...")
    A = np.sum((ar_data == 1) & (precip_data == 1), axis=0)
    B = np.sum((ar_data == 1) & (precip_data == 0), axis=0)
    C = np.sum((ar_data == 0) & (precip_data == 1), axis=0)
    D = np.sum((ar_data == 0) & (precip_data == 0), axis=0)

    del ar_data, precip_data

    p_no_ar = np.where((A + B + C + D) > 0, (C + D) / (A + B + C + D), np.nan)
    frequency_e = np.where((A + B + C + D) > 0, (A + C) / (A + B + C + D), np.nan)
    p_ext_ar = np.where((A + B) > 0, A / (A + B), np.nan)
    p_ext_no_ar = np.where((C + D) > 0, C / (C + D), np.nan)
    paf = np.where(frequency_e > 0, (frequency_e - p_no_ar * p_ext_no_ar) / frequency_e, np.nan)
    af = np.where(p_ext_ar > 0, (p_ext_ar - p_ext_no_ar) / p_ext_ar, np.nan)
    lift = np.where((p_ext_no_ar > 0), p_ext_ar / p_ext_no_ar, np.nan)
    lift_upper_bound = np.nanpercentile(lift, lift_cap_percentile)
    lift = np.where(lift > lift_upper_bound, lift_upper_bound, lift)

    # --- Parallelized Fisher Exact Test ---
    print("Running Fisher Exact Test (Parallelized)...")
    p_sig = np.full(A.shape, np.nan)
    coords = [(i, j, A[i, j], B[i, j], C[i, j], D[i, j])
              for i in range(A.shape[0]) for j in range(A.shape[1])]

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(
            tqdm(pool.imap(fisher_worker, coords), total=len(coords), desc=f"Fisher Test for {analysis_name}"))

    for i, j, p in results:
        p_sig[i, j] = p

    # --- Spatial Smoothing ---
    if spatial_smoothing_window_size > 1:
        print(f"Applying spatial smoothing with window size {spatial_smoothing_window_size}...")
        p_ext_ar = uniform_filter(p_ext_ar, size=spatial_smoothing_window_size, mode='nearest')
        p_ext_no_ar = uniform_filter(p_ext_no_ar, size=spatial_smoothing_window_size, mode='nearest')
        lift = uniform_filter(lift, size=spatial_smoothing_window_size, mode='nearest')
        p_sig = uniform_filter(p_sig, size=spatial_smoothing_window_size, mode='nearest')

    # --- Generate Plots ---
    print("Generating plots...")

    # Create a boolean mask for significant areas
    sig_mask_bool = p_sig < sig_level

    # Create masked data arrays for plotting only significant regions
    p_ext_ar_sig = np.where(sig_mask_bool, p_ext_ar, np.nan)
    p_ext_no_ar_sig = np.where(sig_mask_bool, p_ext_no_ar, np.nan)
    paf_sig = np.where(sig_mask_bool, paf, np.nan)
    af_sig = np.where(sig_mask_bool, af, np.nan)
    lift_sig = np.where(sig_mask_bool, lift, np.nan)

    # Now, call plot_map with the masked data.
    # The sig_mask parameter is no longer needed since we are only plotting significant data.
    plot_map(p_ext_ar_sig, 'P(Extreme | AR) (Significant Areas)', 'p_ext_ar_sig', output_dir, analysis_name,
             sig_mask=None, lons=precip_lons, lats=precip_lats, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min,
             lat_max=lat_max, date_range_str=date_range_str)

    plot_map(p_ext_no_ar_sig, 'P(Extreme | No AR) (Significant Areas)', 'p_ext_no_ar_sig', output_dir, analysis_name,
             sig_mask=None, lons=precip_lons, lats=precip_lats, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min,
             lat_max=lat_max, date_range_str=date_range_str)

    plot_map(paf_sig, "PAF (Significant Areas)", "paf_sig", output_dir, analysis_name, sig_mask=None,
             cmap="coolwarm", lons=precip_lons, lats=precip_lats, lon_min=lon_min, lon_max=lon_max,
             lat_min=lat_min, lat_max=lat_max, date_range_str=date_range_str)

    plot_map(af_sig, "AF (Significant Areas)", "af_sig", output_dir, analysis_name, vmin=0,
             sig_mask=None, cmap="coolwarm", lons=precip_lons, lats=precip_lats, lon_min=lon_min,
             lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, date_range_str=date_range_str)

    plot_map(lift_sig, 'Lift (Significant Areas)', 'lift_sig', output_dir, analysis_name,
             sig_mask=None, lons=precip_lons, lats=precip_lats, lon_min=lon_min, lon_max=lon_max,
             lat_min=lat_min, lat_max=lat_max, date_range_str=date_range_str)

    # --- Summary Statistics ---
    prop_ar_extreme = np.where((A + C) > 0, A / (A + C), np.nan)
    mean_prop_ar_extreme = np.nanmean(prop_ar_extreme)

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/summary_stats_{analysis_name}.txt", "w") as f:
        f.write(f"--- Summary Statistics for {analysis_name} ---\n")
        f.write(f"Date Range: {date_range_str}\n\n")

        metrics = {"P(Extreme | AR)": p_ext_ar, "P(Extreme | No AR)": p_ext_no_ar,
                   "AF": af, "PAF": paf, "Lift": lift}

        for name, data in metrics.items():
            f.write(f"--- {name} ---\n")
            f.write(f"  Mean: {np.nanmean(data):.4f}\n")
            f.write(f"  Min:  {np.nanmin(data):.4f}\n")
            f.write(f"  Max:  {np.nanmax(data):.4f}\n\n")

        f.write("--- Other Stats ---\n")
        f.write(f"Proportion of Extreme Precipitation Events with AR: {mean_prop_ar_extreme:.4f}\n")
        f.write(f"Effective grid points: {np.sum(~np.isnan(lift))}\n")
        f.write(f"Significant grid points (p < {sig_level}): {np.sum(p_sig < sig_level)}\n")

    print(f"Analysis for {analysis_name} complete. Output saved to: {output_dir}")


if __name__ == "__main__":
    # --- Data Loading and Time Matching ---
    print("Reading NetCDF files...")
    try:
        ar_ds = nc.Dataset('G:/ar_analysis/ar_happen.nc', 'r')
        precip_ds = nc.Dataset('G:/ar_analysis/data/extreme_precipitation.nc', 'r')
    except FileNotFoundError as e:
        print(f"Error: One of the input files was not found. {e}")
        exit()

    ar_time = ar_ds.variables['time'][:]
    ar_dates = num2date(ar_time, units=ar_ds.variables['time'].units, calendar=ar_ds.variables['time'].calendar)
    precip_time = precip_ds.variables['time'][:]
    precip_dates = num2date(precip_time, units=precip_ds.variables['time'].units,
                            calendar=precip_ds.variables['time'].calendar)

    # --- User Input and Time Indexing ---
    try:
        start_date_input = input("起始日期（YYYY-MM-DD HH:MM:SS）: ")
        end_date_input = input("结束日期（YYYY-MM-DD HH:MM:SS）: ")
        start_date = datetime.strptime(start_date_input, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(end_date_input, "%Y-%m-%d %H:%M:%S")
        date_range_str_for_output = f"{start_date_input} to {end_date_input}"
    except ValueError:
        print("日期格式不正确，请使用 YYYY-MM-DD HH:MM:SS 格式。")
        exit()

    print("Matching dates and finding indices...")
    precip_date_to_idx = {date: idx for idx, date in enumerate(precip_dates)}

    all_ar_indices = []
    all_precip_indices = []
    common_dates = []

    # Find common dates within the selected range
    for idx, ar_date in enumerate(ar_dates):
        if start_date <= ar_date <= end_date:
            if ar_date in precip_date_to_idx:
                all_ar_indices.append(idx)
                all_precip_indices.append(precip_date_to_idx[ar_date])
                common_dates.append(ar_date)

    all_ar_indices = np.array(all_ar_indices)
    all_precip_indices = np.array(all_precip_indices)
    common_dates = np.array(common_dates)

    if len(common_dates) == 0:
        print("没有共同时间！")
        exit()

    print(f"共同时间步长：{len(common_dates)}")

    precip_lats_all = precip_ds.variables['latitude'][:]
    precip_lons_all = precip_ds.variables['longitude'][:]
    ar_lats_all = ar_ds.variables['lat'][:]
    ar_lons_all = ar_ds.variables['lon'][:]

    # --- REGIONAL ANALYSIS ---
    for region_key, region_props in regions.items():
        lon_min, lon_max = region_props["lon_min"], region_props["lon_max"]
        lat_min, lat_max = region_props["lat_min"], region_props["lat_max"]

        # Find spatial indices for the current region for both datasets
        precip_lat_indices_region = np.where((precip_lats_all >= lat_min) & (precip_lats_all <= lat_max))[0]
        precip_lon_indices_region = np.where((precip_lons_all >= lon_min) & (precip_lons_all <= lon_max))[0]
        ar_lat_indices_region = np.where((ar_lats_all >= lat_min) & (ar_lats_all <= lat_max))[0]
        ar_lon_indices_region = np.where((ar_lons_all >= lon_min) & (ar_lons_all <= lon_max))[0]

        precip_lats_region = precip_lats_all[precip_lat_indices_region]
        precip_lons_region = precip_lons_all[precip_lon_indices_region]

        # Handle potential latitude inversions for this region, mirroring original script logic
        if precip_lats_region.size > 1 and precip_lats_region[0] > precip_lats_region[-1]:
            precip_lats_region = precip_lats_region[::-1]
            precip_lat_indices_region = precip_lat_indices_region[::-1]

        ar_lats_region = ar_lats_all[ar_lat_indices_region]
        if ar_lats_region.size > 1 and ar_lats_region[0] > ar_lats_region[-1]:
            ar_lat_indices_region = ar_lat_indices_region[::-1]

        region_output_dir = os.path.join(base_output_dir, region_props['name'])

        run_analysis(all_ar_indices, all_precip_indices,
                     ar_lat_indices_region, ar_lon_indices_region, precip_lat_indices_region, precip_lon_indices_region,
                     precip_lats_region, precip_lons_region, ar_ds, precip_ds,
                     region_output_dir, region_props['name'], lon_min, lon_max, lat_min, lat_max,
                     date_range_str_for_output)

    # --- SEASONAL ANALYSIS (GLOBAL SCOPE) ---
    print("\n--- Starting Seasonal Analysis (Global Scope) ---")
    global_region = regions["All_East_Asia"]
    lon_min_g, lon_max_g = global_region["lon_min"], global_region["lon_max"]
    lat_min_g, lat_max_g = global_region["lat_min"], global_region["lat_max"]

    # Indices for the global scope
    precip_lat_indices_global = np.where((precip_lats_all >= lat_min_g) & (precip_lats_all <= lat_max_g))[0]
    precip_lon_indices_global = np.where((precip_lons_all >= lon_min_g) & (precip_lons_all <= lon_max_g))[0]
    ar_lat_indices_global = np.where((ar_lats_all >= lat_min_g) & (ar_lats_all <= lat_max_g))[0]
    ar_lon_indices_global = np.where((ar_lons_all >= lon_min_g) & (ar_lons_all <= lon_max_g))[0]

    precip_lats_global = precip_lats_all[precip_lat_indices_global]
    precip_lons_global = precip_lons_all[precip_lon_indices_global]

    # Handle potential latitude inversions for the global scope
    if precip_lats_global.size > 1 and precip_lats_global[0] > precip_lats_global[-1]:
        precip_lats_global = precip_lats_global[::-1]
        precip_lat_indices_global = precip_lat_indices_global[::-1]

    ar_lats_global = ar_lats_all[ar_lat_indices_global]
    if ar_lats_global.size > 1 and ar_lats_global[0] > ar_lats_global[-1]:
        ar_lat_indices_global = ar_lat_indices_global[::-1]

    for season_name, months in seasons.items():
        seasonal_mask = np.array([date.month in months for date in common_dates])

        seasonal_ar_indices = all_ar_indices[seasonal_mask]
        seasonal_precip_indices = all_precip_indices[seasonal_mask]

        season_output_dir = os.path.join(base_output_dir, "seasonal", season_name)

        run_analysis(seasonal_ar_indices, seasonal_precip_indices,
                     ar_lat_indices_global, ar_lon_indices_global, precip_lat_indices_global, precip_lon_indices_global,
                     precip_lats_global, precip_lons_global, ar_ds, precip_ds,
                     season_output_dir, f"Global_{season_name}", lon_min_g, lon_max_g, lat_min_g, lat_max_g,
                     date_range_str_for_output)

    print("\nAll analyses finished.")
    ar_ds.close()
    precip_ds.close()

