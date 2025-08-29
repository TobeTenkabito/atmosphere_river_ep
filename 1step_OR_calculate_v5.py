import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend for saving figures
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from netCDF4 import num2date, date2num
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import fisher_exact, binomtest
from scipy.ndimage import uniform_filter
from numpy.random import default_rng
import os
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed

# =========================
# Adjustable Parameters (Please modify here)
# =========================
# Time period and season
season_months = None  # e.g., [5,6,7] for summer; None for no filtering
# Case-crossover settings
lags = [1]  # Exposure lag days (can be multiple, e.g., [0,1,2])
control_mode = "sample"  # "sample" or "all"
controls_per_case = 5  # Number of controls to sample per case (if control_mode="sample")
random_seed = 42  # Random seed for sampling
# Statistics and significance
sig_level = 0.05
fdr_control = True  # Whether to apply Benjamini–Hochberg FDR control (for grid-cell p-values)

# --- Performance Control ---
# Set to True to use NVIDIA GPU (requires CuPy). Falls back to CPU if False or CuPy is not found.
USE_GPU = True
# Number of CPU cores to use (only when USE_GPU=False). -1 means use all.
N_JOBS = -1
# Process this many longitude points at a time to save memory.
spatial_chunk_size = 50

# Spatial smoothing
grid_aggregate_size = 1  # If >1, apply a moving window average (e.g., 5 for 5x5)
# Placebo tests
placebo_time_shifts = [7, 14, 21, 30, -7, -14, -21, -30]  # Positive/negative days
placebo_month_shuffle_iterations = 20  # Number of iterations for monthly shuffling

# Visualization
lift_cap_percentile = 95
lift_max_val = 10
plot_projection = ccrs.PlateCarree()
# Output
output_dir = "./ar_analysis_output_1"
os.makedirs(output_dir, exist_ok=True)

# === Regional Analysis Switch and Definitions ===
# Set to True to run analysis for all defined regions, False to run only on the full East Asia domain.
RUN_REGIONAL_ANALYSIS = True

# Sub-regions to analyze within the full East Asia domain
regions = {
    "All_East_Asia": {"lon_min": 100, "lon_max": 150, "lat_min": 10, "lat_max": 60, "zh_name": "整个东亚"},
    "South_China": {"lon_min": 105, "lon_max": 125, "lat_min": 18, "lat_max": 32, "zh_name": "华南"},
    "North_China": {"lon_min": 105, "lon_max": 125, "lat_min": 32, "lat_max": 42, "zh_name": "华北"},
    "Korea_Japan_Archipelago": {"lon_min": 125, "lon_max": 145, "lat_min": 30, "lat_max": 45,
                                "zh_name": "朝鲜和日本列岛"},
    "Northeast_Asia": {"lon_min": 120, "lon_max": 140, "lat_min": 40, "lat_max": 58, "zh_name": "东北亚"}
}
# =========================

if USE_GPU:
    try:
        import cupy as cp

        xp = cp
        print("[INFO] CuPy found. Running in GPU mode.")
    except ImportError:
        print("[WARNING] CuPy not found. Falling back to CPU mode.")
        xp = np
        USE_GPU = False
else:
    xp = np
    print("[INFO] Running in CPU mode.")

# --- Initialization ---
rng = default_rng(random_seed)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    ar_ds.close();
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

# ========== Load Full Time-Series Data ==========
print("[INFO] Loading full AR time-series for temporal shifting...")
AR_full_time = ar_ds.variables['ar_happen'][ar_indices, :, :][:, ar_lat_indices_full, :][:, :,
               ar_lon_indices_full].astype(np.int8)
# We load precip data later per chunk to save memory.

# ========== Seasonal Filtering (Optional) ==========
if season_months is not None:
    season_mask = np.isin(months, np.array(season_months, dtype=int))
    AR_full_time = AR_full_time[season_mask, :, :]
    years = years[season_mask]
    months = months[season_mask]
    common_datetimes = common_datetimes[season_mask]
    precip_indices = precip_indices[season_mask]
    n_time = AR_full_time.shape[0]
    print(f"[INFO] Filtered for season {season_months}: {n_time} time steps remain.")


# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------
def shift_in_time(arr, k):
    arr = xp.asarray(arr, dtype=xp.int8)
    if k == 0:
        return arr.copy()
    n_time_local = arr.shape[0]
    result = xp.zeros_like(arr, dtype=xp.int8)
    if k > 0:
        result[k:] = arr[:-k]
        result[:k] = 0  # Fill with 0
    elif k < 0:
        k = -k
        result[:-k] = arr[k:]
        result[-k:] = 0  # Fill with 0
    return result


def month_shuffle(arr, months_vec, rng_local):
    arr_np = arr.get() if hasattr(arr, 'get') else arr
    arr_shuf = arr_np.copy()
    for m in np.unique(months_vec):
        idx = np.where(months_vec == m)[0]
        if idx.size > 1:
            rng_local.shuffle(arr_shuf[idx, ...])
    return arr_shuf


def benjamini_hochberg(pvals, alpha=0.05):
    p_flat = pvals.flatten()
    valid_mask = ~np.isnan(p_flat)
    p_valid = p_flat[valid_mask]
    if len(p_valid) == 0: return np.zeros_like(pvals, dtype=bool)
    n = len(p_valid)
    order = np.argsort(p_valid)
    ranks = np.empty_like(order);
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


def plot_map(data, lons, lats, title, filename, vmin=None, vmax=None, sig_mask=None, cmap='coolwarm'):
    data_cpu = data.get() if hasattr(data, 'get') else data
    sig_mask_cpu = sig_mask.get() if hasattr(sig_mask, 'get') else sig_mask

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=plot_projection)
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=plot_projection)
    ax.add_feature(cfeature.LAND, facecolor='#D3D3D3', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#E6F0FA', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False;
    gl.right_labels = False

    plot_data = data_cpu.copy()
    if sig_mask_cpu is not None:
        plot_data[~sig_mask_cpu] = np.nan

    mesh = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, vmin=vmin, vmax=vmax, transform=plot_projection)
    plt.colorbar(mesh, ax=ax, label=title.split('(')[0].strip(), shrink=0.8, extend='both')
    plt.title(f"{title}\n{start_date_input} to {end_date_input}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------
# CORE CALCULATION FUNCTIONS (CPU/GPU)
# --------------------------------------------------------
def process_grid_cell_cc_cpu(E_cell, A_cell, months_vec, valid_mask, n_time_local, control_mode, controls_per_case,
                             rng_local):
    n10_cell, n01_cell = 0, 0
    E = E_cell.astype(bool) & valid_mask
    A = A_cell.astype(bool) & valid_mask
    case_indices = np.where(E)[0]
    if case_indices.size == 0: return n10_cell, n01_cell
    for t_case in case_indices:
        m = months_vec[t_case]
        ctrl_candidates = np.where((months_vec == m) & (~E) & (np.arange(n_time_local) != t_case))[0]
        if ctrl_candidates.size == 0: continue
        if control_mode == "sample":
            size = controls_per_case
            replace = ctrl_candidates.size < size
            chosen_indices = rng_local.choice(ctrl_candidates, size=size, replace=replace)
        else:
            chosen_indices = ctrl_candidates
        case_exp = A[t_case]
        ctrl_exp = A[chosen_indices]
        n10_cell += np.sum((case_exp == 1) & (ctrl_exp == 0))
        n01_cell += np.sum((case_exp == 0) & (ctrl_exp == 1))
    return n10_cell, n01_cell


def process_grid_cell_fisher_cpu(AR_cell, EXT_cell):
    a = np.sum((AR_cell == 1) & (EXT_cell == 1))
    b = np.sum((AR_cell == 1) & (EXT_cell == 0))
    c = np.sum((AR_cell == 0) & (EXT_cell == 1))
    d = np.sum((AR_cell == 0) & (EXT_cell == 0))
    table = [[a, b], [c, d]]
    if (a + b + c + d) > 0:
        _, p = fisher_exact(table)
        return a, b, c, d, p
    return 0, 0, 0, 0, np.nan


# --------------------------------------------------------
# NEW: Regional Analysis Function
# --------------------------------------------------------
def run_regional_analysis(region_name, region_dict,
                          full_lats, full_lons,
                          ar_lats_all, ar_lons_all,
                          ar_indices, precip_indices,
                          AR_full_time, months):
    print(f"\n\n{'=' * 20}\n[RUNNING ANALYSIS FOR] {region_dict['zh_name']} ({region_name})\n{'=' * 20}")

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

    # --- Subset data for the region ---
    AR_region_time = AR_full_time[:, lat_indices_region, :][:, :, lon_indices_region]

    # --- 1) Case-Crossover: Grid-cell level McNemar / Conditional OR ---
    results_by_lag = {}
    for L in lags:
        print(f"\n[STEP 1] Running Case-Crossover for lag={L} day(s)")
        AR_lag_region = shift_in_time(xp.asarray(AR_region_time), -L)
        valid_time_mask = xp.ones(n_time, dtype=bool)
        if L > 0: valid_time_mask[:L] = False

        n10 = np.zeros((Y, X), dtype=np.int32)
        n01 = np.zeros((Y, X), dtype=np.int32)

        pbar = tqdm(range(0, X, spatial_chunk_size), desc=f"Processing grid chunks (lag={L}, {region_name})")
        for j_start in pbar:
            j_end = min(j_start + spatial_chunk_size, X)
            chunk_lon_indices_full = lon_indices_full[lon_indices_region[j_start:j_end]]

            if 'extreme_precipitation_flag_ocean' in precip_ds.variables:
                precip_ocean_chunk = precip_ds.variables['extreme_precipitation_flag_ocean'][precip_indices, :, :][:,
                                     lat_indices_full, :][:, :, chunk_lon_indices_full]
                precip_land_chunk = precip_ds.variables['extreme_precipitation_flag_land'][precip_indices, :, :][:,
                                    lat_indices_full, :][:, :, chunk_lon_indices_full]
                EXT_chunk_np = np.maximum(precip_ocean_chunk, precip_land_chunk).astype(np.int8)
            else:
                EXT_chunk_np = precip_ds.variables['extreme_precipitation_flag'][precip_indices, :, :][:,
                               lat_indices_full, :][:, :, chunk_lon_indices_full].astype(np.int8)

            EXT_chunk_np_region = EXT_chunk_np[:, lat_indices_region, :]

            if USE_GPU:
                AR_lag_chunk = AR_lag_region[:, :, j_start:j_end]
                EXT_chunk = xp.asarray(EXT_chunk_np_region)

                for i in range(Y):
                    for j in range(EXT_chunk.shape[2]):
                        n10_cell, n01_cell = process_grid_cell_cc_cpu(
                            EXT_chunk[:, i, j].get(), AR_lag_chunk[:, i, j].get(), months, valid_time_mask.get(),
                            n_time,
                            control_mode, controls_per_case, rng
                        )
                        n10[i, j_start + j] = n10_cell
                        n01[i, j_start + j] = n01_cell
            else:
                AR_lag_chunk = AR_lag_region[:, :, j_start:j_end]
                tasks = [
                    delayed(process_grid_cell_cc_cpu)(
                        EXT_chunk_np_region[:, i, j], AR_lag_chunk[:, i, j], months, valid_time_mask, n_time,
                        control_mode,
                        controls_per_case, rng
                    ) for i in range(Y) for j in range(EXT_chunk_np_region.shape[2])
                ]
                results = Parallel(n_jobs=N_JOBS, backend='threading')(tasks)
                results_arr = np.array(results).reshape((Y, EXT_chunk_np_region.shape[2], 2))
                n10[:, j_start:j_end] = results_arr[:, :, 0]
                n01[:, j_start:j_end] = results_arr[:, :, 1]

        OR_grid = (n10 + 0.5) / (n01 + 0.5)
        p_grid = np.full((Y, X), np.nan)
        discordant_pairs = n10 + n01
        valid_mask_cpu = discordant_pairs > 0
        p_values = np.array([binomtest(k, n=n, p=0.5).pvalue
                             for k, n in zip(n10[valid_mask_cpu], discordant_pairs[valid_mask_cpu])])
        p_grid[valid_mask_cpu] = p_values
        sig_mask = benjamini_hochberg(p_grid, alpha=sig_level) if fdr_control else (p_grid < sig_level)
        if grid_aggregate_size > 1:
            OR_grid = uniform_filter(OR_grid, size=grid_aggregate_size, mode='constant', cval=np.nan)

        total_n10, total_n01 = np.sum(n10), np.sum(n01)
        pooled_OR = (total_n10 + 0.5) / (total_n01 + 0.5) if (total_n10 + total_n01) > 0 else np.nan
        pooled_p = binomtest(total_n10, n=(total_n10 + total_n01), p=0.5).pvalue if (
                                                                                                total_n10 + total_n01) > 0 else np.nan

        print(f"[RESULT lag={L}] Pooled OR = {pooled_OR:.3f} (p={pooled_p:.3e})")
        print(f"[RESULT lag={L}] Significant grids (FDR<{sig_level}): {np.sum(sig_mask)}/{Y * X}")

        results_by_lag[L] = {"OR_grid": OR_grid, "p_grid": p_grid, "sig_mask": sig_mask, "pooled_OR": pooled_OR,
                             "pooled_p": pooled_p}
        vmax = np.nanpercentile(OR_grid[sig_mask], 98) if np.any(sig_mask) else 3.0
        plot_map(OR_grid, lons_region, lats_region, f"Conditional OR ({region_dict['zh_name']}, lag={L}d)",
                 f"cc_or_lag{L}_{region_name}", vmin=0.5, vmax=vmax, sig_mask=sig_mask, cmap='plasma')

    # --- 2) Placebo Tests (Time-shift & Month-shuffle) ---
    def calculate_pooled_or_placebo(AR_placebo, lag):
        AR_lag = shift_in_time(xp.asarray(AR_placebo), -lag)
        valid_time_mask = xp.ones(n_time, dtype=bool)
        if lag > 0: valid_time_mask[:lag] = False
        total_n10, total_n01 = 0, 0
        valid_time_mask_cpu = valid_time_mask.get() if USE_GPU else valid_time_mask
        for j_start in range(0, X, spatial_chunk_size):
            j_end = min(j_start + spatial_chunk_size, X)
            chunk_lon_indices_full = lon_indices_full[lon_indices_region[j_start:j_end]]

            if 'extreme_precipitation_flag_ocean' in precip_ds.variables:
                precip_ocean_chunk = precip_ds.variables['extreme_precipitation_flag_ocean'][precip_indices, :, :][:,
                                     lat_indices_full, :][:, :, chunk_lon_indices_full]
                precip_land_chunk = precip_ds.variables['extreme_precipitation_flag_land'][precip_indices, :, :][:,
                                    lat_indices_full, :][:, :, chunk_lon_indices_full]
                EXT_chunk_np = np.maximum(precip_ocean_chunk, precip_land_chunk).astype(np.int8)
            else:
                EXT_chunk_np = precip_ds.variables['extreme_precipitation_flag'][precip_indices, :, :][:,
                               lat_indices_full, :][:, :, chunk_lon_indices_full].astype(np.int8)
            EXT_chunk_np_region = EXT_chunk_np[:, lat_indices_region, :]

            AR_lag_chunk = AR_lag[:, :, j_start:j_end]
            AR_lag_chunk_cpu = AR_lag_chunk.get() if USE_GPU else AR_lag_chunk
            results = Parallel(n_jobs=-1, backend="loky")(
                delayed(process_grid_cell_cc_cpu)(
                    EXT_chunk_np_region[:, i, j],
                    AR_lag_chunk_cpu[:, i, j],
                    months, valid_time_mask_cpu,
                    n_time, control_mode, controls_per_case, rng
                )
                for i in range(Y) for j in range(EXT_chunk_np_region.shape[2])
            )
            n10_sum, n01_sum = map(sum, zip(*results))
            total_n10 += n10_sum
            total_n01 += n01_sum
        return (total_n10 + 0.5) / (total_n01 + 0.5) if (total_n10 + total_n01) > 0 else np.nan

    for L in lags:
        print(f"\n[STEP 2a] Running Placebo Test (Time-shift) for lag={L}, {region_name}")
        real_OR = results_by_lag[L]["pooled_OR"]
        shift_ORs = [calculate_pooled_or_placebo(shift_in_time(AR_region_time, s), L) for s in
                     tqdm(placebo_time_shifts, desc="Time-shifting")]
        shift_ORs = np.array(shift_ORs)
        mc_p = (np.sum(shift_ORs >= real_OR) + 1) / (np.sum(~np.isnan(shift_ORs)) + 1)
        print(
            f"[PLACEBO shift, lag={L}] Real OR={real_OR:.3f}, Placebo Median OR={np.nanmedian(shift_ORs):.3f}, p≈{mc_p:.3f}")

        plt.figure(figsize=(7, 5))
        plt.hist(shift_ORs[~np.isnan(shift_ORs)], bins=10, alpha=0.7, label='Placebo Distribution')
        plt.axvline(real_OR, color='red', linestyle='--', linewidth=2, label=f'Real OR = {real_OR:.2f}')
        plt.xlabel("Pooled OR");
        plt.ylabel("Frequency");
        plt.title(f"Placebo (Time-shift) vs Real OR ({region_dict['zh_name']}, lag={L}d)")
        plt.legend();
        plt.tight_layout();
        plt.savefig(f"{output_dir}/placebo_shift_lag{L}_{region_name}.png", dpi=300);
        plt.close()

        print(f"\n[STEP 2b] Running Placebo Test (Month-shuffle) for lag={L}, {region_name}")
        perm_ORs = [calculate_pooled_or_placebo(month_shuffle(AR_region_time, months, rng), L) for _ in
                    tqdm(range(placebo_month_shuffle_iterations), desc="Month-shuffling")]
        perm_ORs = np.array(perm_ORs)
        mc_p = (np.sum(perm_ORs >= real_OR) + 1) / (np.sum(~np.isnan(perm_ORs)) + 1)
        print(
            f"[PLACEBO shuffle, lag={L}] Real OR={real_OR:.3f}, Placebo Median OR={np.nanmedian(perm_ORs):.3f}, p≈{mc_p:.3f}")

        plt.figure(figsize=(7, 5))
        plt.hist(perm_ORs[~np.isnan(perm_ORs)], bins=20, alpha=0.7, label='Placebo Distribution')
        plt.axvline(real_OR, color='red', linestyle='--', linewidth=2, label=f'Real OR = {real_OR:.2f}')
        plt.xlabel("Pooled OR");
        plt.ylabel("Frequency");
        plt.title(f"Placebo (Month-shuffle) vs Real OR ({region_dict['zh_name']}, lag={L}d)")
        plt.legend();
        plt.tight_layout();
        plt.savefig(f"{output_dir}/placebo_shuffle_lag{L}_{region_name}.png", dpi=300);
        plt.close()

    # --- 3) Traditional Contingency Table + Fisher's Test ---
    print("\n[STEP 3] Calculating traditional contingency tables (Fisher's Exact Test)")
    A = np.zeros((Y, X), dtype=np.int32)
    B = np.zeros((Y, X), dtype=np.int32)
    C = np.zeros((Y, X), dtype=np.int32)
    D = np.zeros((Y, X), dtype=np.int32)
    p_sig_fisher = np.full(A.shape, np.nan)

    pbar = tqdm(range(0, X, spatial_chunk_size), desc=f"Processing grid chunks (Fisher, {region_name})")
    for j_start in pbar:
        j_end = min(j_start + spatial_chunk_size, X)
        chunk_lon_indices_full = lon_indices_full[lon_indices_region[j_start:j_end]]
        ar_chunk_lon_indices_full = ar_lon_indices_full[lon_indices_full.tolist().index(
            chunk_lon_indices_full[0]):lon_indices_full.tolist().index(chunk_lon_indices_full[-1]) + 1]

        AR_chunk_np = ar_ds.variables['ar_happen'][ar_indices, :, :][:, ar_lat_indices_full, :][:, :,
                      ar_chunk_lon_indices_full].astype(np.int8)
        if 'extreme_precipitation_flag_ocean' in precip_ds.variables:
            precip_ocean_chunk = precip_ds.variables['extreme_precipitation_flag_ocean'][precip_indices, :, :][:,
                                 lat_indices_full, :][:, :, chunk_lon_indices_full]
            precip_land_chunk = precip_ds.variables['extreme_precipitation_flag_land'][precip_indices, :, :][:,
                                lat_indices_full, :][:, :, chunk_lon_indices_full]
            EXT_chunk_np = np.maximum(precip_ocean_chunk, precip_land_chunk).astype(np.int8)
        else:
            EXT_chunk_np = precip_ds.variables['extreme_precipitation_flag'][precip_indices, :, :][:, lat_indices_full,
                           :][:, :, chunk_lon_indices_full].astype(np.int8)

        EXT_chunk_np_region = EXT_chunk_np[:, lat_indices_region, :]
        AR_chunk_np_region = AR_chunk_np[:, lat_indices_region, :]

        if season_months is not None:
            AR_chunk_np_region = AR_chunk_np_region[season_mask, :, :]
            EXT_chunk_np_region = EXT_chunk_np_region[season_mask, :, :]

        if USE_GPU:
            AR_chunk = xp.asarray(AR_chunk_np_region)
            EXT_chunk = xp.asarray(EXT_chunk_np_region)

            A_gpu = xp.sum((AR_chunk == 1) & (EXT_chunk == 1), axis=0)
            B_gpu = xp.sum((AR_chunk == 1) & (EXT_chunk == 0), axis=0)
            C_gpu = xp.sum((AR_chunk == 0) & (EXT_chunk == 1), axis=0)
            D_gpu = xp.sum((AR_chunk == 0) & (EXT_chunk == 0), axis=0)

            A[:, j_start:j_end] = A_gpu.get()
            B[:, j_start:j_end] = B_gpu.get()
            C[:, j_start:j_end] = C_gpu.get()
            D[:, j_start:j_end] = D_gpu.get()

            for i in range(Y):
                for j in range(A_gpu.shape[1]):
                    table = [[A[i, j_start + j], B[i, j_start + j]], [C[i, j_start + j], D[i, j_start + j]]]
                    if np.sum(table) > 0:
                        _, p = fisher_exact(table)
                        p_sig_fisher[i, j_start + j] = p
        else:
            tasks = [
                delayed(process_grid_cell_fisher_cpu)(AR_chunk_np_region[:, i, j], EXT_chunk_np_region[:, i, j])
                for i in range(Y) for j in range(EXT_chunk_np_region.shape[2])
            ]
            results = Parallel(n_jobs=N_JOBS, backend='threading')(tasks)
            results_arr = np.array(results).reshape((Y, EXT_chunk_np_region.shape[2], 5))
            A[:, j_start:j_end] = results_arr[:, :, 0]
            B[:, j_start:j_end] = results_arr[:, :, 1]
            C[:, j_start:j_end] = results_arr[:, :, 2]
            D[:, j_start:j_end] = results_arr[:, :, 3]
            p_sig_fisher[:, j_start:j_end] = results_arr[:, :, 4]

    p_ext_ar = np.where((A + B) > 0, A / (A + B), np.nan)
    p_ext_no_ar = np.where((C + D) > 0, C / (C + D), np.nan)
    lift = np.divide(p_ext_ar, p_ext_no_ar, out=np.full_like(p_ext_ar, np.nan), where=p_ext_no_ar > 0)
    lift = np.clip(lift, None, lift_max_val)

    sig_mask_fisher = p_sig_fisher < sig_level
    plot_map(p_ext_ar, lons_region, lats_region, f"P(E|AR) ({region_dict['zh_name']})", f"p_ext_ar_{region_name}",
             sig_mask=sig_mask_fisher, cmap="winter")
    plot_map(p_ext_no_ar, lons_region, lats_region, f"P(E|no AR) ({region_dict['zh_name']})",
             f"p_ext_no_ar_{region_name}", sig_mask=sig_mask_fisher, cmap="winter")
    plot_map(lift, lons_region, lats_region, f"Lift [P(E|AR)/P(E|no AR)] ({region_dict['zh_name']})",
             f"lift_sig_{region_name}", vmin=0.5, vmax=np.nanpercentile(lift, lift_cap_percentile),
             sig_mask=sig_mask_fisher, cmap='plasma')

    # --- Final Summary Output ---
    with open(f"{output_dir}/summary_analysis_{region_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"Analysis Summary for {region_dict['zh_name']}\n{'=' * 20}\n")
        f.write(f"Time window: {start_date_input} to {end_date_input}\n")
        f.write(f"Season months: {'All' if season_months is None else season_months}\n")
        f.write(
            f"Spatial Domain: Lon({lons_region.min()}-{lons_region.max()}), Lat({lats_region.min()}-{lats_region.max()})\n\n")
        f.write(f"Case-Crossover Analysis\n{'-' * 20}\n")
        f.write(f"Parameters: lags={lags}, control_mode={control_mode}, controls_per_case={controls_per_case}\n")
        for L in lags:
            res = results_by_lag[L]
            f.write(f"  Lag={L}d: Pooled OR = {res['pooled_OR']:.4f} (p={res['pooled_p']:.3e})\n")
            f.write(f"         Significant grids (FDR<{sig_level}): {np.sum(res['sig_mask'])} / {Y * X}\n")
        f.write(f"\nTraditional Contingency Analysis (Fisher's Exact)\n{'-' * 20}\n")
        f.write(f"  Mean P(Extreme|AR) = {np.nanmean(p_ext_ar):.4f}\n")
        f.write(f"  Mean P(Extreme|No AR) = {np.nanmean(p_ext_no_ar):.4f}\n")
        f.write(f"  Mean Lift = {np.nanmean(lift):.4f}\n")
        f.write(f"  Significant grids (p<{sig_level}): {np.sum(sig_mask_fisher)} / {Y * X}\n")

    print(f"[DONE] Analysis for {region_name} is complete. Outputs are saved in: {output_dir}")


# --------------------------------------------------------
# Main Execution Flow
# --------------------------------------------------------
if RUN_REGIONAL_ANALYSIS:
    for region_name, region_dict in regions.items():
        run_regional_analysis(region_name, region_dict,
                              lats_full, lons_full,
                              ar_lats_all, ar_lons_all,
                              ar_indices, precip_indices,
                              AR_full_time, months)
else:
    run_regional_analysis("All_East_Asia", regions["All_East_Asia"],
                          lats_full, lons_full,
                          ar_lats_all, ar_lons_all,
                          ar_indices, precip_indices,
                          AR_full_time, months)

# Close datasets
ar_ds.close()
precip_ds.close()
