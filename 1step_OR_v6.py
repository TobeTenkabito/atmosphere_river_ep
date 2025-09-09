import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving figures
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from netCDF4 import num2date, date2num
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import fisher_exact, binomtest, norm
from scipy.ndimage import uniform_filter
from numpy.random import default_rng
import os
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogFormatterSciNotation
warnings.filterwarnings('ignore')
# =========================
# Adjustable Parameters (Please modify here)
# =========================
# Time period and season
season_months = None  # e.g., [5,6,7] for summer; None for no filtering
# Verification switch
Time_shifting = True
Month_shuffling = True
# Case-crossover settings
lags = [1]  # Exposure lag days (can be multiple, e.g., [0,1,2])
control_mode = "sample"  # "sample" or "all"
controls_per_case = 5  # Number of controls to sample per case (if control_mode="sample")
random_seed = 42  # Random seed for sampling
# Statistics and significance
sig_level = 0.05
fdr_control = True  # Whether to apply Benjamini–Hochberg FDR control (for grid-cell p-values)

# --- Performance Control ---
# Number of CPU cores to use for parallel processing. -1 means use all available cores.
N_JOBS = -1
# Process this many longitude points at a time to save memory during initial data load.
# Note: With sparse matrices, memory usage is much lower, so this can often be larger.
spatial_chunk_size = 100

# Spatial smoothing
grid_aggregate_size = 1  # If >1, apply a moving window average (e.g., 5 for 5x5)
# Placebo tests
placebo_time_shifts = [7, 14, 21, 30, -7, -14, -21, -30]  # Positive/negative days
placebo_month_shuffle_iterations = 20  # Number of iterations for monthly shuffling

# Visualization
lift_cap_percentile = 95
lift_max_val = 100
plot_projection = ccrs.PlateCarree()
# Output
output_dir = "./analysis_output"
os.makedirs(output_dir, exist_ok=True)

# === Regional Analysis Switch and Definitions ===
# Set to True to run analysis for all defined regions, False to run only on the full East Asia domain.
RUN_REGIONAL_ANALYSIS = True

# Sub-regions to analyze within the full East Asia domain
regions = {
    "All_East_Asia": {"lon_min": 100, "lon_max": 150, "lat_min": 10, "lat_max": 60, "name": "East_Asia"},
    "South_China": {"lon_min": 105, "lon_max": 125, "lat_min": 18, "lat_max": 35, "name": "South_China"},
    "North_China": {"lon_min": 105, "lon_max": 125, "lat_min": 32, "lat_max": 45, "name": "North_China"},
    "Korea_Japan_Archipelago": {"lon_min": 122, "lon_max": 146, "lat_min": 30, "lat_max": 50, "name": "Korea_Japan"},
    "Northeast_Asia": {"lon_min": 120, "lon_max": 140, "lat_min": 40, "lat_max": 60, "name": "Northeast_Asia"}
}
# =========================

# --- Initialization ---
rng = default_rng(random_seed)
warnings.filterwarnings("ignore", category=RuntimeWarning)
xp = np  # Using NumPy as the primary array library. Sparse operations handle efficiency.

# ========== Dataset Paths ==========
ar_ds_path = 'G:/ar_analysis/ar_happen.nc'
precip_ds_path = 'G:/ar_analysis/data/extreme_precipitation.nc'

if not (os.path.exists(ar_ds_path) and os.path.exists(precip_ds_path)):
    print(f"[ERROR] Cannot find input files. Please check paths:\n- {ar_ds_path}\n- {precip_ds_path}")
    raise SystemExit

ar_ds = nc.Dataset(ar_ds_path)
precip_ds = nc.Dataset(precip_ds_path)

# ========== Time Reading ==========
ar_time = ar_ds.variables['time'][:]
ar_time_units = ar_ds.variables['time'].units
ar_time_calendar = getattr(ar_ds.variables['time'], 'calendar', 'standard')
ar_dates_all = num2date(ar_time, units=ar_time_units, calendar=ar_time_calendar)

precip_time = precip_ds.variables['time'][:]
precip_time_units = precip_ds.variables['time'].units
precip_time_calendar = getattr(precip_ds.variables['time'], 'calendar', 'standard')
precip_dates_all = num2date(precip_time, units=precip_time_units, calendar=precip_time_calendar)

print(f"[INFO] AR time range: {ar_dates_all[0]} to {ar_dates_all[-1]}")
print(f"[INFO] Precip time range: {precip_dates_all[0]} to {precip_dates_all[-1]}")

# ========== User Input for Time Period ==========
start_date_input = input("Enter start date (YYYY-MM-DD HH:MM:SS): ").strip()
end_date_input = input("Enter end date (YYYY-MM-DD HH:MM:SS): ").strip()
start_date = datetime.strptime(start_date_input, "%Y-%m-%d %H:%M:%S")
end_date = datetime.strptime(end_date_input, "%Y-%m-%d %H:%M:%S")

# ========== Time Alignment ==========
start_time_ar = date2num(start_date, ar_time_units, ar_time_calendar)
end_time_ar = date2num(end_date, ar_time_units, ar_time_calendar)
start_time_precip = date2num(start_date, precip_time_units, precip_time_calendar)
end_time_precip = date2num(end_date, precip_time_units, precip_time_calendar)
ar_idx = (ar_time >= start_time_ar) & (ar_time <= end_time_ar)
precip_idx = (precip_time >= start_time_precip) & (precip_time <= end_time_precip)

ar_dates_sel = np.array(ar_dates_all)[ar_idx]
precip_dates_sel = np.array(precip_dates_all)[precip_idx]

unified_units = "hours since 1970-01-01 00:00:00"
calendar = "proleptic_gregorian"
ar_time_unified = np.round(date2num(ar_dates_sel, unified_units, calendar), 6)
precip_time_unified = np.round(date2num(precip_dates_sel, unified_units, calendar), 6)
common_times, ar_comm_indices, precip_comm_indices = np.intersect1d(
    ar_time_unified, precip_time_unified, return_indices=True
)

if len(common_times) == 0:
    print("[ERROR] No common time steps found between the datasets for the selected period.")
    ar_ds.close()
    precip_ds.close()
    raise SystemExit

ar_indices = np.where(ar_idx)[0][ar_comm_indices]
precip_indices = np.where(precip_idx)[0][precip_comm_indices]
n_time = len(common_times)

common_datetimes = num2date(common_times, units=unified_units, calendar=calendar)
years = np.array([dt.year for dt in common_datetimes])
months = np.array([dt.month for dt in common_datetimes])

print(f"[INFO] Found {n_time} common time steps.")
print(f"[INFO] Common time range: {common_datetimes[0]} to {common_datetimes[-1]}")

# ========== Spatial Cropping for Full East Asia Domain ==========
full_lon_min, full_lon_max = 100, 150
full_lat_min, full_lat_max = 10, 60

precip_lats_all = precip_ds.variables['latitude'][:]
precip_lons_all = precip_ds.variables['longitude'][:]
lat_mask_full = (precip_lats_all >= full_lat_min) & (precip_lats_all <= full_lat_max)
lon_mask_full = (precip_lons_all >= full_lon_min) & (precip_lons_all <= full_lon_max)
lat_indices_full = np.where(lat_mask_full)[0]
lon_indices_full = np.where(lon_mask_full)[0]
lats_full = precip_lats_all[lat_indices_full]
lons_full = precip_lons_all[lon_indices_full]

if lats_full[0] > lats_full[-1]:
    lats_full = lats_full[::-1]
    lat_indices_full = lat_indices_full[::-1]

ar_lats_all = ar_ds.variables['lat'][:]
ar_lons_all = ar_ds.variables['lon'][:]
ar_lat_mask_full = (ar_lats_all >= full_lat_min) & (ar_lats_all <= full_lat_max)
ar_lon_mask_full = (ar_lons_all >= full_lon_min) & (ar_lons_all <= full_lon_max)
ar_lat_indices_full = np.where(ar_lat_mask_full)[0]
ar_lon_indices_full = np.where(ar_lon_mask_full)[0]

if not (np.allclose(ar_lats_all[ar_lat_indices_full], lats_full) and np.allclose(ar_lons_all[ar_lon_indices_full],
                                                                                 lons_full)):
    print("[ERROR] AR and Precipitation spatial grids are not consistent after cropping!")
    ar_ds.close();
    precip_ds.close()
    raise SystemExit

Y_full, X_full = len(lats_full), len(lons_full)
print(f"[INFO] Full spatial grid defined: {Y_full} latitudes, {X_full} longitudes.")

# ========== Seasonal Filtering (Optional) ==========
season_mask = np.ones(n_time, dtype=bool)
if season_months is not None:
    season_mask = np.isin(months, np.array(season_months, dtype=int))
    years = years[season_mask]
    months = months[season_mask]
    common_datetimes = common_datetimes[season_mask]
    precip_indices = precip_indices[season_mask]
    ar_indices = ar_indices[season_mask]
    n_time = np.sum(season_mask)
    print(f"[INFO] Filtered for season {season_months}: {n_time} time steps remain.")


# --------------------------------------------------------
# Helper and Calculation Functions
# --------------------------------------------------------
def shift_in_time(arr, k):
    """Shifts a 2D sparse matrix in time (axis 0)."""
    if k == 0:
        return arr.copy()

    n_time_local = arr.shape[0]
    rows, cols = arr.nonzero()

    if k > 0:  # Shift forward, data from past moves to present
        new_rows = rows + k
        mask = new_rows < n_time_local
    elif k < 0:  # Shift backward
        k_abs = -k
        new_rows = rows - k_abs
        mask = new_rows >= 0

    return csc_matrix((arr.data[mask], (new_rows[mask], cols[mask])), shape=arr.shape, dtype=np.int8)


def month_shuffle(arr, months_vec, rng_local):
    """Shuffles a sparse matrix within each month."""
    arr_coo = arr.tocoo()
    shuffled_rows = arr_coo.row.copy()
    for m in np.unique(months_vec):
        month_indices = np.where(months_vec == m)[0]
        if len(month_indices) > 1:
            # Find which data points fall in this month
            data_in_month_mask = np.isin(arr_coo.row, month_indices)

            # Map original time indices to new shuffled positions within the month
            shuffled_month_indices = month_indices.copy()
            rng_local.shuffle(shuffled_month_indices)

            time_map = {orig: shuffled for orig, shuffled in zip(month_indices, shuffled_month_indices)}

            # Apply the mapping to the rows of the data points
            original_rows_in_month = arr_coo.row[data_in_month_mask]
            shuffled_rows[data_in_month_mask] = np.vectorize(time_map.get)(original_rows_in_month)

    return csc_matrix((arr_coo.data, (shuffled_rows, arr_coo.col)), shape=arr.shape, dtype=np.int8)


def benjamini_hochberg(pvals, alpha=0.05):
    """Performs the Benjamini-Hochberg FDR correction on a 2D array of p-values."""
    p_flat = pvals.flatten()
    valid_mask = ~np.isnan(p_flat)
    p_valid = p_flat[valid_mask]
    if len(p_valid) == 0: return np.zeros_like(pvals, dtype=bool)
    n = len(p_valid)
    order = np.argsort(p_valid)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    crit_vals = (ranks / n) * alpha
    passed = p_valid[order] <= crit_vals
    sig_mask_flat = np.zeros_like(p_flat, dtype=bool)
    if np.any(passed):
        k_max = np.max(np.where(passed)[0])
        p_threshold = p_valid[order][k_max]
        sig_mask_valid = p_valid <= p_threshold
        sig_mask_flat[valid_mask] = sig_mask_valid
    return sig_mask_flat.reshape(pvals.shape)


def process_grid_cell_cc(E_cell, A_cell, months_vec, valid_time_mask, n_time_local, control_mode, controls_per_case,
                         rng_local):
    """Processes a single grid cell for case-crossover. Operates on dense vectors."""
    n10_cell, n01_cell = 0, 0
    E = E_cell.astype(bool) & valid_time_mask
    A = A_cell.astype(bool) & valid_time_mask
    case_indices = np.where(E)[0]

    if case_indices.size == 0: return n10_cell, n01_cell

    for t_case in case_indices:
        m = months_vec[t_case]
        ctrl_candidates = np.where((months_vec == m) & (~E) & (np.arange(n_time_local) != t_case))[0]
        if ctrl_candidates.size == 0: continue

        if control_mode == "sample":
            size = min(controls_per_case, ctrl_candidates.size)
            chosen_indices = rng_local.choice(ctrl_candidates, size=size, replace=False)
        else:
            chosen_indices = ctrl_candidates

        case_exp = A[t_case]
        ctrl_exp = A[chosen_indices]

        n10_cell += np.sum((case_exp == 1) & (ctrl_exp == 0))
        n01_cell += np.sum((case_exp == 0) & (ctrl_exp == 1))

    return n10_cell, n01_cell


def calculate_fisher_stats_sparse(AR_sparse, EXT_sparse):
    """Calculates contingency table stats and lift using vectorized sparse matrix operations."""
    print("[INFO] Calculating contingency tables with sparse operations...")

    # Ensure matrices are boolean for logical operations
    AR_bool = AR_sparse.astype(bool)
    EXT_bool = EXT_sparse.astype(bool)

    # A: AR=1, EXT=1
    A = AR_bool.multiply(EXT_bool).sum(axis=0).A1
    # B: AR=1, EXT=0
    B = AR_bool.sum(axis=0).A1 - A
    # C: AR=0, EXT=1
    C = EXT_bool.sum(axis=0).A1 - A
    # D: Total - A - B - C
    D = AR_sparse.shape[0] - (A + B + C)
    data = {'A': A, 'B': B, 'C': C, 'D': D}
    # create DataFrame
    df = pd.DataFrame(data)
    df.to_csv('ABCD.csv', index=False)

    # Calculate probabilities and lift
    with np.errstate(divide='ignore', invalid='ignore'):
        p_ext_ar = A / (A + B)
        p_ext_no_ar = C / (C + D)
        lift = p_ext_ar / p_ext_no_ar

    p_ext_ar[np.isinf(p_ext_ar) | np.isnan(p_ext_ar)] = np.nan
    p_ext_no_ar[np.isinf(p_ext_no_ar) | np.isnan(p_ext_no_ar)] = np.nan
    lift[np.isinf(lift) | np.isnan(lift)] = np.nan
    lift = np.clip(lift, None, lift_max_val)

    # Calculate 95% CI for Lift using log method
    # SE(log(Lift)) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    with np.errstate(divide='ignore', invalid='ignore'):
        var_log_lift = (1 / A - 1 / (A + B)) + (1 / C - 1 / (C + D))
        se_log_lift = np.sqrt(var_log_lift)

        log_lift = np.log(lift)
        z = norm.ppf(1 - sig_level / 2)  # ~1.96

        log_ci_lower = log_lift - z * se_log_lift
        log_ci_upper = log_lift + z * se_log_lift

        lift_ci_lower = np.exp(log_ci_lower)
        lift_ci_upper = np.exp(log_ci_upper)

    # Run Fisher's exact test in parallel for p-values
    print("[INFO] Running Fisher's Exact Test for significance...")
    tasks = [delayed(fisher_exact)([[a, b], [c, d]]) for a, b, c, d in zip(A, B, C, D)]
    results = Parallel(n_jobs=N_JOBS)(tqdm(tasks, desc="Fisher's Tests"))
    p_values_fisher = np.array([res[1] for res in results])

    return {
        "p_ext_ar": p_ext_ar, "p_ext_no_ar": p_ext_no_ar,
        "lift": lift, "lift_ci_lower": lift_ci_lower, "lift_ci_upper": lift_ci_upper,
        "p_values_fisher": p_values_fisher
    }


# --------------------------------------------------------
# Plotting Functions
# --------------------------------------------------------
def create_custom_cmap(colors, name='custom_cmap'):
    """
    Creates a custom colormap from a list of colors.

    Args:
        colors (list): A list of colors (e.g., ['blue', 'white', 'red']).
        name (str): The name for the custom colormap.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The custom colormap object.
    """
    return LinearSegmentedColormap.from_list(name, colors)


def plot_map(data, lons, lats, title, filename, vmin=None, vmax=None, sig_mask=None, cmap='coolwarm', n_levels=7,
             extend='both'):
    """Generic map plotting function with adaptive color scaling and discrete levels."""
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#D3D3D3', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#E6F0FA', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Adaptive vmin and vmax calculation
    if vmin is None or vmax is None:
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            if vmin is None:
                vmin = np.nanpercentile(valid_data, 1)
            if vmax is None:
                vmax = np.nanpercentile(valid_data, 99)
        else:
            vmin, vmax = 0, 1

    plot_data = data.copy()
    if sig_mask is not None:
        plot_data[~sig_mask] = np.nan

    # Create discrete levels based on vmin, vmax, and n_levels
    levels = np.linspace(vmin, vmax, n_levels)

    mesh = ax.contourf(lons, lats, plot_data, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
    plt.colorbar(mesh, ax=ax, label=title.split('(')[0].strip(), shrink=0.8, extend=extend)
    plt.title(f"{title}\n{start_date_input} to {end_date_input}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_ratio_map(data, lons, lats, title, filename, sig_mask=None, cmap='coolwarm', n_levels=10, extend='both'):
    """
    Plots a map for ratio-based data (e.g., Odds Ratio, Lift) with a
    symmetrical color scale centered around 1.0. The range is determined by percentiles
    to avoid issues with extreme outliers.
    """
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#D3D3D3', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#E6F0FA', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    valid_data = data[~np.isnan(data) & (data > 0)]
    if len(valid_data) > 0:
        log10_data = np.log10(valid_data)

        # Calculate percentiles to create a robust, symmetrical range
        p_min_log = np.nanpercentile(log10_data, 1)
        p_max_log = np.nanpercentile(log10_data, 99)

        # Use the larger of the two absolute percentile values for symmetry
        max_deviation = max(np.abs(p_min_log), np.abs(p_max_log))

        # Add a fallback for cases where the data is too clustered.
        # This ensures a minimum range for visual clarity.
        robust_range_threshold = 0.5  # Corresponds to a linear range of approx [0.3, 3.1]
        if max_deviation < robust_range_threshold:
            max_deviation = robust_range_threshold

        vmin_log = -max_deviation
        vmax_log = max_deviation
        levels = np.linspace(vmin_log, vmax_log, n_levels)
    else:
        vmin_log, vmax_log = 0, 1
        levels = np.linspace(vmin_log, vmax_log, n_levels)

    plot_data = data.copy()
    plot_data = np.log10(plot_data)
    if sig_mask is not None:
        plot_data[~sig_mask] = np.nan

    mesh = ax.contourf(lons, lats, plot_data, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())

    # Generate log-spaced ticks and format them
    num_ticks = 7
    vmin_linear = 10 ** vmin_log
    vmax_linear = 10 ** vmax_log

    if n_levels > num_ticks:
        tick_levels = np.logspace(np.log10(vmin_linear), np.log10(vmax_linear), num_ticks)
    else:
        tick_levels = np.logspace(np.log10(vmin_linear), np.log10(vmax_linear), n_levels)

    # Use a logarithmic colorbar formatter
    cbar_formatter = LogFormatterSciNotation(base=10, labelOnlyBase=False)
    cbar = plt.colorbar(mesh, ax=ax, label=title.split('(')[0].strip(), shrink=0.8, extend=extend, ticks=tick_levels,
                        format=cbar_formatter)
    cbar.set_label(title.split('(')[0].strip(), fontsize=10)

    plt.title(f"{title}\n{start_date_input} to {end_date_input}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_pooled_or_ci(results_by_lag, region_name, region_name_1):
    """Plots a bar chart of pooled Odds Ratios with 95% CI error bars with adaptive y-axis."""
    lags_list = list(results_by_lag.keys())
    pooled_ors = [results_by_lag[L]["pooled_OR"] for L in lags_list]
    ci_lower = [results_by_lag[L]["pooled_OR_ci_lower"] for L in lags_list]
    ci_upper = [results_by_lag[L]["pooled_OR_ci_upper"] for L in lags_list]

    errors = [np.array(pooled_ors) - np.array(ci_lower), np.array(ci_upper) - np.array(pooled_ors)]

    plt.figure(figsize=(8, 6))
    plt.bar(lags_list, pooled_ors, yerr=errors, capsize=5, color='skyblue', edgecolor='black')
    plt.axhline(1, color='red', linestyle='--', linewidth=1)

    # Adaptive y-axis limits
    y_min = np.min(ci_lower) if ci_lower else 0.5
    y_max = np.max(ci_upper) if ci_upper else 1.5

    # Add a buffer for better visualization
    y_range = y_max - y_min
    y_min_lim = max(0, y_min - y_range * 0.1)
    y_max_lim = y_max + y_range * 0.1

    # Ensure the y-axis includes the y=1 line
    y_min_lim = min(y_min_lim, 0.9)
    y_max_lim = max(y_max_lim, 1.1)

    plt.ylim(y_min_lim, y_max_lim)

    plt.xlabel("Lag (days)")
    plt.ylabel("Pooled Odds Ratio (95% CI)")
    plt.title(f"Pooled OR for {region_name_1}")
    plt.xticks(lags_list)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pooled_or_ci_{region_name}.png", dpi=300)
    plt.close()


# --------------------------------------------------------
# Main Regional Analysis Function
# --------------------------------------------------------
def run_regional_analysis(region_name, region_dict,
                          full_lats, full_lons,
                          ar_lat_indices_full, ar_lon_indices_full,
                          lat_indices_full, lon_indices_full,
                          ar_indices, precip_indices,
                          months):
    print(f"\n\n{'=' * 20}\n[RUNNING ANALYSIS FOR] {region_dict['name']} ({region_name})\n{'=' * 20}")

    lon_min, lon_max = region_dict['lon_min'], region_dict['lon_max']
    lat_min, lat_max = region_dict['lat_min'], region_dict['lat_max']

    # --- Find regional indices ---
    lat_mask_region = (full_lats >= lat_min) & (full_lats <= lat_max)
    lon_mask_region = (full_lons >= lon_min) & (full_lons <= lon_max)
    lat_indices_region = np.where(lat_mask_region)[0]
    lon_indices_region = np.where(lon_mask_region)[0]
    lats_region = full_lats[lat_indices_region]
    lons_region = full_lons[lon_indices_region]

    if len(lats_region) == 0 or len(lons_region) == 0:
        print(f"[WARNING] No grid points found for region: {region_name}. Skipping.")
        return

    Y, X = len(lats_region), len(lons_region)
    print(f"[INFO] Regional grid defined: {Y} latitudes, {X} longitudes.")

    # --- Load regional data into sparse matrices ---
    print("[INFO] Loading regional data into sparse matrices...")

    # Get the correct slice indices for the netCDF files
    ar_regional_lat_slice = ar_lat_indices_full[lat_indices_region]
    ar_regional_lon_slice = ar_lon_indices_full[lon_indices_region]
    precip_regional_lat_slice = lat_indices_full[lat_indices_region]
    precip_regional_lon_slice = lon_indices_full[lon_indices_region]

    AR_region_dense = ar_ds.variables['ar_happen'][ar_indices, :, :][:, ar_regional_lat_slice, :][:, :,
                      ar_regional_lon_slice].astype(np.int8)
    AR_region_sparse = csc_matrix(AR_region_dense.reshape(n_time, Y * X))

    if 'extreme_precipitation_flag_ocean' in precip_ds.variables:
        precip_ocean = precip_ds.variables['extreme_precipitation_flag_ocean'][precip_indices, :, :][:,
                       precip_regional_lat_slice, :][:, :, precip_regional_lon_slice]
        precip_land = precip_ds.variables['extreme_precipitation_flag_land'][precip_indices, :, :][:,
                      precip_regional_lat_slice, :][:, :, precip_regional_lon_slice]
        EXT_region_dense = np.maximum(precip_ocean, precip_land).astype(np.int8)
    else:
        EXT_region_dense = precip_ds.variables['extreme_precipitation_flag'][precip_indices, :, :][:,
                           precip_regional_lat_slice, :][:, :, precip_regional_lon_slice].astype(np.int8)

    EXT_region_sparse = csc_matrix(EXT_region_dense.reshape(n_time, Y * X))
    print(
        f"[INFO] Sparse matrix densities: AR={AR_region_sparse.nnz / np.prod(AR_region_sparse.shape):.2%}, Precip={EXT_region_sparse.nnz / np.prod(EXT_region_sparse.shape):.2%}")

    # --- 1) Case-Crossover: Grid-cell level ---
    results_by_lag = {}
    for L in lags:
        print(f"\n[STEP 1] Running Case-Crossover for lag={L} day(s)")
        AR_lag_sparse = shift_in_time(AR_region_sparse, -L)
        valid_time_mask = np.ones(n_time, dtype=bool)
        if L > 0: valid_time_mask[:L] = False

        # Parallel processing per grid cell
        tasks = [
            delayed(process_grid_cell_cc)(
                EXT_region_sparse[:, i].toarray().flatten(),
                AR_lag_sparse[:, i].toarray().flatten(),
                months, valid_time_mask, n_time, control_mode, controls_per_case, rng
            ) for i in range(Y * X)
        ]

        results = Parallel(n_jobs=N_JOBS)(tqdm(tasks, desc=f"CC Analysis (lag={L})"))
        results_arr = np.array(results)
        n10 = results_arr[:, 0].reshape(Y, X)
        n01 = results_arr[:, 1].reshape(Y, X)

        # Calculate OR and p-values
        with np.errstate(divide='ignore', invalid='ignore'):
            OR_grid = (n10 + 0.5) / (n01 + 0.5)

            # Calculate 95% CI for OR using log method
            log_OR = np.log(OR_grid)
            se_log_OR = np.sqrt(1 / (n10 + 0.5) + 1 / (n01 + 0.5))
            z = norm.ppf(1 - sig_level / 2)  # ~1.96

            OR_ci_lower = np.exp(log_OR - z * se_log_OR)
            OR_ci_upper = np.exp(log_OR + z * se_log_OR)

        p_grid = np.full((Y, X), np.nan)
        discordant_pairs = n10 + n01
        valid_mask_cpu = discordant_pairs > 0
        p_values_list = [binomtest(k, n=n, p=0.5).pvalue
                         for k, n in zip(n10[valid_mask_cpu], discordant_pairs[valid_mask_cpu])]
        p_grid[valid_mask_cpu] = p_values_list

        sig_mask = benjamini_hochberg(p_grid, alpha=sig_level) if fdr_control else (p_grid < sig_level)
        if grid_aggregate_size > 1:
            OR_grid = uniform_filter(OR_grid, size=grid_aggregate_size, mode='constant', cval=np.nan)

        # Pooled analysis
        total_n10, total_n01 = np.sum(n10), np.sum(n01)
        pooled_OR = (total_n10 + 0.5) / (total_n01 + 0.5) if (total_n10 + total_n01) > 0 else np.nan
        pooled_p = binomtest(total_n10, n=(total_n10 + total_n01), p=0.5).pvalue if (
                                                                                                total_n10 + total_n01) > 0 else np.nan

        # Pooled CI
        log_pooled_OR = np.log(pooled_OR)
        se_log_pooled_OR = np.sqrt(1 / (total_n10 + 0.5) + 1 / (total_n01 + 0.5))
        pooled_OR_ci_lower = np.exp(log_pooled_OR - z * se_log_pooled_OR)
        pooled_OR_ci_upper = np.exp(log_pooled_OR + z * se_log_pooled_OR)

        print(
            f"[RESULT lag={L}] Pooled OR = {pooled_OR:.3f} (95% CI: {pooled_OR_ci_lower:.3f}-{pooled_OR_ci_upper:.3f}), p={pooled_p:.3e}")
        print(f"[RESULT lag={L}] Significant grids (FDR<{sig_level}): {np.sum(sig_mask)}/{Y * X}")

        results_by_lag[L] = {
            "OR_grid": OR_grid, "p_grid": p_grid, "sig_mask": sig_mask,
            "OR_ci_lower": OR_ci_lower, "OR_ci_upper": OR_ci_upper,
            "pooled_OR": pooled_OR, "pooled_p": pooled_p,
            "pooled_OR_ci_lower": pooled_OR_ci_lower, "pooled_OR_ci_upper": pooled_OR_ci_upper
        }

        # your color styles
        color0 = create_custom_cmap(['lightblue', 'white', 'thistle'])
        color1 = create_custom_cmap(['#66CCFF', '#FF9966'])

        vmax = np.nanpercentile(OR_grid[sig_mask], 98) if np.any(sig_mask) else 3.0
        plot_map(OR_grid, lons_region, lats_region, f"Conditional OR ({region_dict['name']}, lag={L}d)",
                 f"cc_or_ci_lag{L}_{region_name}", vmin=0.5, vmax=vmax, sig_mask=sig_mask, cmap=color1)
        plot_ratio_map(OR_grid, lons_region, lats_region, f"Conditional log OR ({region_dict['name']}, lag={L}d)",
                 f"cc_log_or_lag{L}_{region_name}", sig_mask=sig_mask, cmap=color0)
        plot_map(OR_ci_lower, lons_region, lats_region, f"OR 95% CI Lower Bound ({region_dict['name']}, lag={L}d)",
                 f"cc_or_ci_lower_lag{L}_{region_name}", vmin=0.5, vmax=vmax, sig_mask=sig_mask, cmap='plasma')
        plot_map(OR_ci_upper, lons_region, lats_region, f"OR 95% CI Upper Bound ({region_dict['name']}, lag={L}d)",
                 f"cc_or_ci_upper_lag{L}_{region_name}", vmin=0.5, vmax=vmax, sig_mask=sig_mask, cmap='plasma')

    plot_pooled_or_ci(results_by_lag, region_name, region_dict['name'])

    # --- 2) Placebo Tests (Time-shift & Month-shuffle) ---
    # (This section remains largely the same, but now operates on the in-memory sparse matrices)
    def calculate_pooled_or_placebo(AR_placebo_sparse, lag):
        AR_lag = shift_in_time(AR_placebo_sparse, -lag)
        valid_time_mask = np.ones(n_time, dtype=bool)
        if lag > 0: valid_time_mask[:lag] = False

        tasks = [
            delayed(process_grid_cell_cc)(
                EXT_region_sparse[:, i].toarray().flatten(),
                AR_lag[:, i].toarray().flatten(),
                months, valid_time_mask, n_time, control_mode, controls_per_case, rng
            ) for i in range(Y * X)
        ]
        results = Parallel(n_jobs=N_JOBS)(tasks)  # No tqdm for placebo to reduce clutter
        n10_sum, n01_sum = map(sum, zip(*results))
        return (n10_sum + 0.5) / (n01_sum + 0.5) if (n10_sum + n01_sum) > 0 else np.nan

    for L in lags:
        print(f"\n[STEP 2] Running Placebo Tests for lag={L}, {region_name}")
        real_OR = results_by_lag[L]["pooled_OR"]

        if Time_shifting:
            # Time-shift placebo
            shift_ORs = [calculate_pooled_or_placebo(shift_in_time(AR_region_sparse, s), L) for s in
                         tqdm(placebo_time_shifts, desc="Time-shifting")]
            shift_ORs = np.array(shift_ORs)
            mc_p_shift = (np.sum(shift_ORs >= real_OR) + 1) / (np.sum(~np.isnan(shift_ORs)) + 1)
            print(
                f"[PLACEBO shift, lag={L}] Real OR={real_OR:.3f}, Placebo Median OR={np.nanmedian(shift_ORs):.3f}, p≈{mc_p_shift:.3f}")
        if Month_shuffling:
            # Month-shuffle placebo
            perm_ORs = [calculate_pooled_or_placebo(month_shuffle(AR_region_sparse, months, rng), L) for _ in
                        tqdm(range(placebo_month_shuffle_iterations), desc="Month-shuffling")]
            perm_ORs = np.array(perm_ORs)
            mc_p_perm = (np.sum(perm_ORs >= real_OR) + 1) / (np.sum(~np.isnan(perm_ORs)) + 1)
            print(
                f"[PLACEBO shuffle, lag={L}] Real OR={real_OR:.3f}, Placebo Median OR={np.nanmedian(perm_ORs):.3f}, p≈{mc_p_perm:.3f}")

    # --- 3) Traditional Contingency Table + Fisher's Test ---
    print("\n[STEP 3] Calculating traditional contingency tables (Fisher's Exact Test)")
    fisher_results = calculate_fisher_stats_sparse(AR_region_sparse, EXT_region_sparse)

    # Reshape results back to 2D grid
    for key, val in fisher_results.items():
        fisher_results[key] = val.reshape(Y, X)

    sig_mask_fisher = fisher_results["p_values_fisher"] < sig_level

    # Plotting
    plot_map(fisher_results["p_ext_ar"], lons_region, lats_region, f"P(E|AR) ({region_dict['name']})",
             f"p_ext_ar_{region_name}",
             sig_mask=sig_mask_fisher, cmap="winter", extend='max')

    lift_vmax = np.nanpercentile(fisher_results["lift"], lift_cap_percentile)
    plot_map(fisher_results["lift"], lons_region, lats_region, f"Lift [P(E|AR)/P(E|no AR)] ({region_dict['name']})",
             f"lift_sig_{region_name}", vmin=0.5, vmax=lift_vmax, sig_mask=sig_mask_fisher, cmap='plasma')
    plot_map(fisher_results["lift_ci_lower"], lons_region, lats_region,
             f"Lift 95% CI Lower Bound ({region_dict['name']})",
             f"lift_ci_lower_{region_name}", vmin=0.5, vmax=lift_vmax, sig_mask=sig_mask_fisher, cmap='plasma')
    plot_map(fisher_results["lift_ci_upper"], lons_region, lats_region,
             f"Lift 95% CI Upper Bound ({region_dict['name']})",
             f"lift_ci_upper_{region_name}", vmin=0.5, vmax=lift_vmax, sig_mask=sig_mask_fisher, cmap='plasma')

    # --- Final Summary Output ---
    with open(f"{output_dir}/summary_analysis_{region_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"Analysis Summary for {region_dict['name']}\n{'=' * 20}\n")
        f.write(f"Time window: {start_date_input} to {end_date_input}\n")
        f.write(f"Season months: {'All' if season_months is None else season_months}\n")
        f.write(
            f"Spatial Domain: Lon({lons_region.min()}-{lons_region.max()}), Lat({lats_region.min()}-{lats_region.max()})\n\n")
        f.write(f"Case-Crossover Analysis\n{'-' * 20}\n")
        f.write(f"Parameters: lags={lags}, control_mode={control_mode}, controls_per_case={controls_per_case}\n")
        for L in lags:
            res = results_by_lag[L]
            f.write(
                f"  Lag={L}d: Pooled OR = {res['pooled_OR']:.4f} (95% CI: {res['pooled_OR_ci_lower']:.4f}-{res['pooled_OR_ci_upper']:.4f}, p={res['pooled_p']:.3e})\n")
            f.write(f"         Significant grids (FDR<{sig_level}): {np.sum(res['sig_mask'])} / {Y * X}\n")
        f.write(f"\nTraditional Contingency Analysis (Fisher's Exact)\n{'-' * 20}\n")
        f.write(f"  Mean P(Extreme|AR) = {np.nanmean(fisher_results['p_ext_ar']):.4f}\n")
        f.write(f"  Mean P(Extreme|No AR) = {np.nanmean(fisher_results['p_ext_no_ar']):.4f}\n")
        f.write(f"  Mean Lift = {np.nanmean(fisher_results['lift']):.4f}\n")
        f.write(f"  Significant grids (p<{sig_level}): {np.sum(sig_mask_fisher)} / {Y * X}\n")

    print(f"[DONE] Analysis for {region_name} is complete. Outputs are saved in: {output_dir}")


# --------------------------------------------------------
# Main Execution Flow
# --------------------------------------------------------
if __name__ == "__main__":
    if RUN_REGIONAL_ANALYSIS:
        run_regional_analysis("South_China", regions["South_China"],
                              lats_full, lons_full,
                              ar_lat_indices_full, ar_lon_indices_full,
                              lat_indices_full, lon_indices_full,
                              ar_indices, precip_indices,
                              months)
        run_regional_analysis("North_China", regions["North_China"],
                              lats_full, lons_full,
                              ar_lat_indices_full, ar_lon_indices_full,
                              lat_indices_full, lon_indices_full,
                              ar_indices, precip_indices,
                              months)
        run_regional_analysis("Korea_Japan_Archipelago", regions["Korea_Japan_Archipelago"],
                              lats_full, lons_full,
                              ar_lat_indices_full, ar_lon_indices_full,
                              lat_indices_full, lon_indices_full,
                              ar_indices, precip_indices,
                              months)
        run_regional_analysis("Northeast_Asia", regions["Northeast_Asia"],
                              lats_full, lons_full,
                              ar_lat_indices_full, ar_lon_indices_full,
                              lat_indices_full, lon_indices_full,
                              ar_indices, precip_indices,
                              months)
    else:
        run_regional_analysis("All_East_Asia", regions["All_East_Asia"],
                              lats_full, lons_full,
                              ar_lat_indices_full, ar_lon_indices_full,
                              lat_indices_full, lon_indices_full,
                              ar_indices, precip_indices,
                              months)
    # Close datasets
    ar_ds.close()
    precip_ds.close()

