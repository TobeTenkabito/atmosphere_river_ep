import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from netCDF4 import num2date
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import fisher_exact, norm
from scipy.ndimage import uniform_filter
from tqdm import tqdm
import multiprocessing as mp
import os
import gc
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# Configuration
# ============================================================

spatial_smoothing_window_size = 1
sig_level = 0.05
lift_cap_percentile = 95
base_output_dir = "G:/ar_analysis/output_seasonal_regional"

AR_FILE = "./ar_happen_ERA5.nc"
PRECIP_FILE = "G:/ar_analysis/data/extreme_precipitation.nc"

AR_VAR = "ar_happen"
PRECIP_VAR = "extreme_precipitation_flag"

# 时间分块大小：越大越快但更吃内存
TIME_CHUNK_SIZE = 256

# 空间分块大小：越大越快但更吃内存
LAT_TILE_SIZE = 20
LON_TILE_SIZE = 20

# worker 数量：默认榨干 CPU
N_WORKERS = max(1, mp.cpu_count() - 1)

os.makedirs(base_output_dir, exist_ok=True)


# ============================================================
# Region and Season Definitions
# ============================================================

regions = {
    "All_East_Asia": {
        "lon_min": 100,
        "lon_max": 150,
        "lat_min": 10,
        "lat_max": 60,
        "name": "East_Asia"
    },
    "South_China": {
        "lon_min": 105,
        "lon_max": 125,
        "lat_min": 18,
        "lat_max": 35,
        "name": "South_China"
    },
    "North_China": {
        "lon_min": 105,
        "lon_max": 125,
        "lat_min": 32,
        "lat_max": 45,
        "name": "North_China"
    },
    "Korea_Japan_Archipelago": {
        "lon_min": 122,
        "lon_max": 146,
        "lat_min": 30,
        "lat_max": 50,
        "name": "Korea_Japan"
    },
    "Northeast_Asia": {
        "lon_min": 120,
        "lon_max": 140,
        "lat_min": 40,
        "lat_max": 60,
        "name": "Northeast_Asia"
    }
}

seasons = {
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "DJF": [12, 1, 2]
}


# ============================================================
# Statistics
# ============================================================

def delta_ci(A, B, C, D, alpha=0.05, cc=0.5):
    """
    Compute RR, AF, PAF with CI using delta method from aggregated counts.
    """

    A_c = A + (cc if A == 0 else 0)
    B_c = B + (cc if B == 0 else 0)
    C_c = C + (cc if C == 0 else 0)
    D_c = D + (cc if D == 0 else 0)

    n1 = A_c + B_c
    n0 = C_c + D_c

    if n1 == 0 or n0 == 0:
        return {
            'RR': np.nan,
            'RR_lo': np.nan,
            'RR_hi': np.nan,
            'AF': np.nan,
            'AF_lo': np.nan,
            'AF_hi': np.nan,
            'PAF': np.nan,
            'PAF_lo': np.nan,
            'PAF_hi': np.nan
        }

    p1 = A_c / n1
    p2 = C_c / n0

    if p2 == 0:
        return {
            'RR': np.nan,
            'RR_lo': np.nan,
            'RR_hi': np.nan,
            'AF': np.nan,
            'AF_lo': np.nan,
            'AF_hi': np.nan,
            'PAF': np.nan,
            'PAF_lo': np.nan,
            'PAF_hi': np.nan
        }

    RR = p1 / p2

    var_logRR = (1.0 / A_c) - (1.0 / n1) + (1.0 / C_c) - (1.0 / n0)
    var_logRR = max(var_logRR, 0)

    se_logRR = np.sqrt(var_logRR)
    z = norm.ppf(1 - alpha / 2)

    RR_lo = np.exp(np.log(RR) - z * se_logRR)
    RR_hi = np.exp(np.log(RR) + z * se_logRR)

    AF = 1.0 - 1.0 / RR
    se_AF = se_logRR / RR
    AF_lo = AF - z * se_AF
    AF_hi = AF + z * se_AF

    p_exp = (A + B) / (A + B + C + D) if (A + B + C + D) > 0 else np.nan
    denom = 1.0 + p_exp * (RR - 1.0)

    PAF = (p_exp * (RR - 1.0)) / denom
    deriv = (p_exp * RR) / (denom ** 2) if denom != 0 else 0.0
    se_PAF = se_logRR * deriv

    PAF_lo = PAF - z * se_PAF
    PAF_hi = PAF + z * se_PAF

    return {
        'RR': RR,
        'RR_lo': RR_lo,
        'RR_hi': RR_hi,
        'AF': AF,
        'AF_lo': AF_lo,
        'AF_hi': AF_hi,
        'PAF': PAF,
        'PAF_lo': PAF_lo,
        'PAF_hi': PAF_hi
    }


def fisher_p_grid(A, B, C, D):
    """
    对一个 tile 内的所有格点做 Fisher exact test。
    不生成巨大 coords 列表，直接循环二维数组。
    """

    p = np.full(A.shape, np.nan, dtype=np.float32)

    nlat, nlon = A.shape

    for i in range(nlat):
        for j in range(nlon):
            a = int(A[i, j])
            b = int(B[i, j])
            c = int(C[i, j])
            d = int(D[i, j])

            if a + b + c + d > 0:
                _, pv = fisher_exact([[a, b], [c, d]])
                p[i, j] = pv

    return p


# ============================================================
# Tile Utilities
# ============================================================

def make_tiles(nlat, nlon, lat_tile_size=20, lon_tile_size=20):
    """
    生成空间 tile。
    返回每个 tile 在区域数组中的局部起止位置。
    """

    tiles = []

    for lat_start in range(0, nlat, lat_tile_size):
        lat_end = min(lat_start + lat_tile_size, nlat)

        for lon_start in range(0, nlon, lon_tile_size):
            lon_end = min(lon_start + lon_tile_size, nlon)

            tiles.append((lat_start, lat_end, lon_start, lon_end))

    return tiles


def to_contiguous_blocks(indices):
    """
    把一组索引拆成尽可能连续的块。
    对 NetCDF 来说，连续切片比花式索引快很多、也省内存。
    返回：
        [(start, end, local_positions), ...]
    其中 end 是 Python slice 的 exclusive end。
    """

    indices = np.asarray(indices)

    if len(indices) == 0:
        return []

    blocks = []
    start_pos = 0

    for k in range(1, len(indices)):
        if indices[k] != indices[k - 1] + 1:
            block_indices = indices[start_pos:k]
            blocks.append((block_indices[0], block_indices[-1] + 1, np.arange(start_pos, k)))
            start_pos = k

    block_indices = indices[start_pos:]
    blocks.append((block_indices[0], block_indices[-1] + 1, np.arange(start_pos, len(indices))))

    return blocks


# ============================================================
# Worker
# ============================================================

def compute_tile_worker(args):
    """
    每个 worker 处理一个空间 tile：
    - 自己打开 NetCDF 文件
    - 按时间 chunk 读取
    - 累计 A/B/C/D
    - 计算 Fisher p
    """

    (
        ar_file,
        precip_file,
        ar_var_name,
        precip_var_name,
        ar_time_indices,
        precip_time_indices,
        ar_lat_indices_region,
        ar_lon_indices_region,
        precip_lat_indices_region,
        precip_lon_indices_region,
        tile
    ) = args

    lat_start, lat_end, lon_start, lon_end = tile

    # 当前 tile 对应的真实 NetCDF 空间索引
    ar_lat_tile = ar_lat_indices_region[lat_start:lat_end]
    ar_lon_tile = ar_lon_indices_region[lon_start:lon_end]
    precip_lat_tile = precip_lat_indices_region[lat_start:lat_end]
    precip_lon_tile = precip_lon_indices_region[lon_start:lon_end]

    tile_nlat = lat_end - lat_start
    tile_nlon = lon_end - lon_start

    A = np.zeros((tile_nlat, tile_nlon), dtype=np.uint32)
    B = np.zeros((tile_nlat, tile_nlon), dtype=np.uint32)
    C = np.zeros((tile_nlat, tile_nlon), dtype=np.uint32)
    D = np.zeros((tile_nlat, tile_nlon), dtype=np.uint32)

    # 每个进程自己打开文件，避免多进程共享 Dataset
    ar_ds = nc.Dataset(ar_file, "r")
    precip_ds = nc.Dataset(precip_file, "r")

    ar_var = ar_ds.variables[ar_var_name]
    precip_var = precip_ds.variables[precip_var_name]

    ntime = len(ar_time_indices)

    for t0 in range(0, ntime, TIME_CHUNK_SIZE):
        t1 = min(t0 + TIME_CHUNK_SIZE, ntime)

        ar_t_idx = ar_time_indices[t0:t1]
        precip_t_idx = precip_time_indices[t0:t1]

        # ----------------------------------------------------
        # 注意：
        # 这里假设 ar_time_indices 和 precip_time_indices 在时间上已经一一对应。
        # 为了避免高级索引造成巨大临时数组，这里仍然会用时间索引数组。
        # 如果你的时间索引是连续的，速度会更快。
        # ----------------------------------------------------

        ar_block = ar_var[
            ar_t_idx,
            ar_lat_tile,
            ar_lon_tile
        ].astype(np.int8)

        precip_block = precip_var[
            precip_t_idx,
            precip_lat_tile,
            precip_lon_tile
        ].astype(np.int8)

        A += np.sum((ar_block == 1) & (precip_block == 1), axis=0, dtype=np.uint32)
        B += np.sum((ar_block == 1) & (precip_block == 0), axis=0, dtype=np.uint32)
        C += np.sum((ar_block == 0) & (precip_block == 1), axis=0, dtype=np.uint32)
        D += np.sum((ar_block == 0) & (precip_block == 0), axis=0, dtype=np.uint32)

        del ar_block, precip_block

    ar_ds.close()
    precip_ds.close()

    p_sig = fisher_p_grid(A, B, C, D)

    return lat_start, lat_end, lon_start, lon_end, A, B, C, D, p_sig


# ============================================================
# Metric Calculation from Counts
# ============================================================

def compute_metrics_from_counts(A, B, C, D, p_sig):
    """
    根据 A/B/C/D 计算所有指标。
    """

    total = A + B + C + D

    with np.errstate(divide='ignore', invalid='ignore'):
        p_no_ar = np.where(total > 0, (C + D) / total, np.nan)
        frequency_e = np.where(total > 0, (A + C) / total, np.nan)

        p_ext_ar = np.where((A + B) > 0, A / (A + B), np.nan)
        p_ext_no_ar = np.where((C + D) > 0, C / (C + D), np.nan)

        paf = np.where(
            frequency_e > 0,
            (frequency_e - p_no_ar * p_ext_no_ar) / frequency_e,
            np.nan
        )

        af = np.where(
            p_ext_ar > 0,
            (p_ext_ar - p_ext_no_ar) / p_ext_ar,
            np.nan
        )

        lift = np.where(
            p_ext_no_ar > 0,
            p_ext_ar / p_ext_no_ar,
            np.nan
        )

    if np.any(np.isfinite(lift)):
        lift_upper_bound = np.nanpercentile(lift, lift_cap_percentile)
        lift = np.where(lift > lift_upper_bound, lift_upper_bound, lift)

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

    return metrics


# ============================================================
# Plotting
# ============================================================

def plot_map(
    data,
    title,
    filename,
    output_dir,
    region_name,
    vmin=None,
    vmax=None,
    sig_mask=None,
    cmap='viridis',
    n_levels=7,
    lon_min=100,
    lon_max=150,
    lat_min=10,
    lat_max=60,
    lons=None,
    lats=None,
    date_range_str=""
):
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#D3D3D3')
    ax.add_feature(cfeature.OCEAN, facecolor='#E6F0FA')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAKES, facecolor='#e6f0ff', edgecolor='none', alpha=0.9, zorder=2)
    ax.add_feature(cfeature.RIVERS, edgecolor='#4a90e2', linewidth=0.45, alpha=0.75, zorder=5)

    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    if vmin is None or vmax is None:
        valid_data = data[np.isfinite(data)]

        if len(valid_data) > 0:
            if vmin is None:
                vmin = np.nanmin(valid_data)
            if vmax is None:
                vmax = np.nanmax(valid_data)

            if title.startswith("PAF") or title.startswith("AF"):
                vmin = max(0, vmin)
        else:
            vmin, vmax = 0, 1

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0, 1

    levels = np.linspace(vmin, vmax, n_levels + 1)

    mesh = ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading='auto'
    )

    if np.any(np.isfinite(data)):
        try:
            ax.contour(
                lons,
                lats,
                data,
                levels=levels,
                colors='black',
                linewidths=0.5,
                transform=ccrs.PlateCarree()
            )
        except Exception:
            pass

    if sig_mask is not None:
        significant_mask = np.where(sig_mask < sig_level, 1, np.nan)
        ax.contourf(
            lons,
            lats,
            significant_mask,
            levels=[0.5, 1.5],
            colors='none',
            hatches=['...'],
            transform=ccrs.PlateCarree()
        )

    cbar = plt.colorbar(mesh, ax=ax, label=title, shrink=0.8)
    cbar.set_label(title, fontsize=10)

    full_title = f"{title} for {region_name}\n{date_range_str}"
    plt.title(full_title, fontsize=14)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_seasonal_subfigures(
    seasonal_data,
    output_dir,
    filename,
    title,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    lons,
    lats
):
    all_valid = []

    for d in seasonal_data.values():
        valid = d[np.isfinite(d)]
        if len(valid) > 0:
            all_valid.append(valid)

    if len(all_valid) > 0:
        all_data = np.concatenate(all_valid)
        vmin = np.nanmin(all_data)
        vmax = np.nanmax(all_data)
    else:
        vmin, vmax = 0, 1

    if title in ["PAF", "AF"]:
        vmin = 0

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0, 1

    cmap = 'viridis' if title == "Lift" else 'coolwarm'

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    seasons_list = list(seasons.keys())

    mesh = None

    for i, season_name in enumerate(seasons_list):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        if season_name not in seasonal_data:
            ax.set_title(f"{season_name}: No data")
            continue

        data = seasonal_data[season_name]

        mesh = ax.pcolormesh(
            lons,
            lats,
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading='auto'
        )

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#D3D3D3')
        ax.add_feature(cfeature.OCEAN, facecolor='#E6F0FA')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.LAKES, facecolor='#e6f0ff', edgecolor='none', alpha=0.9, zorder=2)
        ax.add_feature(cfeature.RIVERS, edgecolor='#4a90e2', linewidth=0.45, alpha=0.75, zorder=5)

        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False

        if col != 0:
            gl.left_labels = False
        if row != 1:
            gl.bottom_labels = False

        ax.text(
            0.02,
            0.98,
            subplot_labels[i],
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='left'
        )

        ax.text(
            0.98,
            0.98,
            season_name,
            transform=ax.transAxes,
            fontsize=12,
            va='top',
            ha='right'
        )

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    if mesh is not None:
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(f'{title} Value', fontsize=12)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_seasonal_subfigures(
    seasonal_data_dict,
    output_dir,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    lons,
    lats
):
    metrics_to_plot = {
        "p_ext_ar": "P(Extreme | AR)",
        "p_ext_no_ar": "P(Extreme | No AR)",
        "paf": "PAF",
        "af": "AF",
        "lift": "Lift"
    }

    for metric_key, metric_title in metrics_to_plot.items():
        single_metric_data = {}

        for season in seasons.keys():
            if metric_key in seasonal_data_dict and season in seasonal_data_dict[metric_key]:
                single_metric_data[season] = seasonal_data_dict[metric_key][season]

        if not single_metric_data:
            print(f"Skipping combined plot for {metric_title} due to missing seasonal data.")
            continue

        filename = f"combined_seasonal_{metric_key}"

        print(f"Generating combined seasonal subplot for: {metric_title}")

        plot_seasonal_subfigures(
            single_metric_data,
            output_dir,
            filename,
            metric_title,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            lons,
            lats
        )


# ============================================================
# Main Analysis Function
# ============================================================

def run_analysis_parallel(
    ar_indices,
    precip_indices,
    ar_lat_indices,
    ar_lon_indices,
    precip_lat_indices,
    precip_lon_indices,
    precip_lats,
    precip_lons,
    output_dir,
    analysis_name,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    date_range_str
):
    """
    内存安全并行版本：
    - 空间分 tile
    - 时间分 chunk
    - worker 内独立打开 NetCDF
    - 主进程拼接二维结果
    """

    print(f"\n--- Starting Parallel Memory-Safe Analysis for: {analysis_name} ---")

    if len(ar_indices) == 0:
        print(f"Skipping {analysis_name} due to no common time steps.")
        return None

    nlat = len(precip_lat_indices)
    nlon = len(precip_lon_indices)

    print(f"Grid size for {analysis_name}: {nlat} x {nlon}")
    print(f"Time steps: {len(ar_indices)}")
    print(f"Workers: {N_WORKERS}")
    print(f"Tile size: {LAT_TILE_SIZE} x {LON_TILE_SIZE}")
    print(f"Time chunk size: {TIME_CHUNK_SIZE}")

    A = np.zeros((nlat, nlon), dtype=np.uint32)
    B = np.zeros((nlat, nlon), dtype=np.uint32)
    C = np.zeros((nlat, nlon), dtype=np.uint32)
    D = np.zeros((nlat, nlon), dtype=np.uint32)
    p_sig = np.full((nlat, nlon), np.nan, dtype=np.float32)

    tiles = make_tiles(
        nlat,
        nlon,
        lat_tile_size=LAT_TILE_SIZE,
        lon_tile_size=LON_TILE_SIZE
    )

    print(f"Number of spatial tiles: {len(tiles)}")

    tasks = []

    for tile in tiles:
        tasks.append(
            (
                AR_FILE,
                PRECIP_FILE,
                AR_VAR,
                PRECIP_VAR,
                np.asarray(ar_indices, dtype=np.int64),
                np.asarray(precip_indices, dtype=np.int64),
                np.asarray(ar_lat_indices, dtype=np.int64),
                np.asarray(ar_lon_indices, dtype=np.int64),
                np.asarray(precip_lat_indices, dtype=np.int64),
                np.asarray(precip_lon_indices, dtype=np.int64),
                tile
            )
        )

    # Windows 下建议使用 spawn，避免 netCDF/HDF5 fork 问题
    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=N_WORKERS, maxtasksperchild=10) as pool:
        iterator = pool.imap_unordered(compute_tile_worker, tasks, chunksize=1)

        for result in tqdm(iterator, total=len(tasks), desc=f"Tiles for {analysis_name}"):
            lat_start, lat_end, lon_start, lon_end, At, Bt, Ct, Dt, p_tile = result

            A[lat_start:lat_end, lon_start:lon_end] = At
            B[lat_start:lat_end, lon_start:lon_end] = Bt
            C[lat_start:lat_end, lon_start:lon_end] = Ct
            D[lat_start:lat_end, lon_start:lon_end] = Dt
            p_sig[lat_start:lat_end, lon_start:lon_end] = p_tile

            del At, Bt, Ct, Dt, p_tile

    print("Computing metrics from contingency tables...")

    metrics = compute_metrics_from_counts(A, B, C, D, p_sig)

    print("Generating plots...")

    plot_map(
        metrics["p_ext_ar"],
        'P(Extreme | AR) Significant Areas',
        'p_ext_ar_sig',
        output_dir,
        analysis_name,
        sig_mask=None,
        lons=precip_lons,
        lats=precip_lats,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        date_range_str=date_range_str
    )

    plot_map(
        metrics["p_ext_no_ar"],
        'P(Extreme | No AR) Significant Areas',
        'p_ext_no_ar_sig',
        output_dir,
        analysis_name,
        sig_mask=None,
        lons=precip_lons,
        lats=precip_lats,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        date_range_str=date_range_str
    )

    plot_map(
        metrics["paf"],
        "PAF Significant Areas",
        "paf_sig",
        output_dir,
        analysis_name,
        sig_mask=None,
        cmap="coolwarm",
        lons=precip_lons,
        lats=precip_lats,
        lon_min=lon_min,
        lon_max=lon_max,
        vmin=0,
        lat_min=lat_min,
        lat_max=lat_max,
        date_range_str=date_range_str
    )

    plot_map(
        metrics["af"],
        "AF Significant Areas",
        "af_sig",
        output_dir,
        analysis_name,
        vmin=0,
        sig_mask=None,
        cmap="coolwarm",
        lons=precip_lons,
        lats=precip_lats,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        date_range_str=date_range_str
    )

    plot_map(
        metrics["lift"],
        "Lift Significant Areas",
        "lift_sig",
        output_dir,
        analysis_name,
        sig_mask=None,
        lons=precip_lons,
        lats=precip_lats,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        date_range_str=date_range_str
    )

    # Summary statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        prop_ar_extreme = np.where((A + C) > 0, A / (A + C), np.nan)

    mean_prop_ar_extreme = np.nanmean(prop_ar_extreme)

    A_tot = np.nansum(A)
    B_tot = np.nansum(B)
    C_tot = np.nansum(C)
    D_tot = np.nansum(D)

    ci_res = delta_ci(A_tot, B_tot, C_tot, D_tot)

    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/summary_stats_{analysis_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"--- Summary Statistics for {analysis_name} ---\n")
        f.write(f"Date Range: {date_range_str}\n\n")

        for name, data in metrics.items():
            f.write(f"--- {name} ---\n")

            if np.any(np.isfinite(data)):
                f.write(f"  Mean: {np.nanmean(data):.4f}\n")
                f.write(f"  Min:  {np.nanmin(data):.4f}\n")
                f.write(f"  Max:  {np.nanmax(data):.4f}\n\n")
            else:
                f.write("  Mean: NaN\n")
                f.write("  Min:  NaN\n")
                f.write("  Max:  NaN\n\n")

        f.write("--- Other Stats ---\n")
        f.write(f"Proportion of Extreme Precipitation Events with AR: {mean_prop_ar_extreme:.4f}\n")
        f.write(f"Effective grid points: {np.sum(np.isfinite(metrics['lift']))}\n")
        f.write(f"Significant grid points p < {sig_level}: {np.sum(p_sig < sig_level)}\n\n")

        f.write("--- Delta Method CI ---\n")
        f.write(f"RR  = {ci_res['RR']:.4f} ({ci_res['RR_lo']:.4f}, {ci_res['RR_hi']:.4f})\n")
        f.write(f"AF  = {ci_res['AF']:.4f} ({ci_res['AF_lo']:.4f}, {ci_res['AF_hi']:.4f})\n")
        f.write(f"PAF = {ci_res['PAF']:.4f} ({ci_res['PAF_lo']:.4f}, {ci_res['PAF_hi']:.4f})\n")

    print(f"Analysis for {analysis_name} complete. Output saved to: {output_dir}")

    del A, B, C, D, p_sig
    gc.collect()

    return metrics


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("Reading NetCDF metadata...")

    try:
        ar_ds = nc.Dataset(AR_FILE, 'r')
        precip_ds = nc.Dataset(PRECIP_FILE, 'r')
    except FileNotFoundError as e:
        print(f"Error: One of the input files was not found. {e}")
        exit()

    ar_time = ar_ds.variables['time'][:]
    ar_dates = num2date(
        ar_time,
        units=ar_ds.variables['time'].units,
        calendar=getattr(ar_ds.variables['time'], 'calendar', 'standard')
    )

    precip_time = precip_ds.variables['time'][:]
    precip_dates = num2date(
        precip_time,
        units=precip_ds.variables['time'].units,
        calendar=getattr(precip_ds.variables['time'], 'calendar', 'standard')
    )

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

    for idx, ar_date in enumerate(ar_dates):
        if start_date <= ar_date <= end_date:
            if ar_date in precip_date_to_idx:
                all_ar_indices.append(idx)
                all_precip_indices.append(precip_date_to_idx[ar_date])
                common_dates.append(ar_date)

    all_ar_indices = np.asarray(all_ar_indices, dtype=np.int64)
    all_precip_indices = np.asarray(all_precip_indices, dtype=np.int64)
    common_dates = np.asarray(common_dates)

    if len(common_dates) == 0:
        print("没有共同时间！")
        exit()

    print(f"共同时间步长：{len(common_dates)}")

    precip_lats_all = precip_ds.variables['latitude'][:]
    precip_lons_all = precip_ds.variables['longitude'][:]

    ar_lats_all = ar_ds.variables['lat'][:]
    ar_lons_all = ar_ds.variables['lon'][:]

    ar_ds.close()
    precip_ds.close()

    # ========================================================
    # Regional Analysis
    # ========================================================

    for region_key, region_props in regions.items():

        lon_min = region_props["lon_min"]
        lon_max = region_props["lon_max"]
        lat_min = region_props["lat_min"]
        lat_max = region_props["lat_max"]

        print(f"\nPreparing region: {region_props['name']}")

        precip_lat_indices_region = np.where(
            (precip_lats_all >= lat_min) & (precip_lats_all <= lat_max)
        )[0]

        precip_lon_indices_region = np.where(
            (precip_lons_all >= lon_min) & (precip_lons_all <= lon_max)
        )[0]

        ar_lat_indices_region = np.where(
            (ar_lats_all >= lat_min) & (ar_lats_all <= lat_max)
        )[0]

        ar_lon_indices_region = np.where(
            (ar_lons_all >= lon_min) & (ar_lons_all <= lon_max)
        )[0]

        precip_lats_region = precip_lats_all[precip_lat_indices_region]
        precip_lons_region = precip_lons_all[precip_lon_indices_region]

        if precip_lats_region.size > 1 and precip_lats_region[0] > precip_lats_region[-1]:
            precip_lats_region = precip_lats_region[::-1]
            precip_lat_indices_region = precip_lat_indices_region[::-1]

        ar_lats_region = ar_lats_all[ar_lat_indices_region]

        if ar_lats_region.size > 1 and ar_lats_region[0] > ar_lats_region[-1]:
            ar_lat_indices_region = ar_lat_indices_region[::-1]

        region_output_dir = os.path.join(base_output_dir, region_props['name'])

        run_analysis_parallel(
            all_ar_indices,
            all_precip_indices,
            ar_lat_indices_region,
            ar_lon_indices_region,
            precip_lat_indices_region,
            precip_lon_indices_region,
            precip_lats_region,
            precip_lons_region,
            region_output_dir,
            region_props['name'],
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            date_range_str_for_output
        )

    # ========================================================
    # Seasonal Analysis
    # ========================================================

    print("\n--- Starting Seasonal Analysis Global Scope ---")

    global_region = regions["All_East_Asia"]

    lon_min_g = global_region["lon_min"]
    lon_max_g = global_region["lon_max"]
    lat_min_g = global_region["lat_min"]
    lat_max_g = global_region["lat_max"]

    precip_lat_indices_global = np.where(
        (precip_lats_all >= lat_min_g) & (precip_lats_all <= lat_max_g)
    )[0]

    precip_lon_indices_global = np.where(
        (precip_lons_all >= lon_min_g) & (precip_lons_all <= lon_max_g)
    )[0]

    ar_lat_indices_global = np.where(
        (ar_lats_all >= lat_min_g) & (ar_lats_all <= lat_max_g)
    )[0]

    ar_lon_indices_global = np.where(
        (ar_lons_all >= lon_min_g) & (ar_lons_all <= lon_max_g)
    )[0]

    precip_lats_global = precip_lats_all[precip_lat_indices_global]
    precip_lons_global = precip_lons_all[precip_lon_indices_global]

    if precip_lats_global.size > 1 and precip_lats_global[0] > precip_lats_global[-1]:
        precip_lats_global = precip_lats_global[::-1]
        precip_lat_indices_global = precip_lat_indices_global[::-1]

    ar_lats_global = ar_lats_all[ar_lat_indices_global]

    if ar_lats_global.size > 1 and ar_lats_global[0] > ar_lats_global[-1]:
        ar_lat_indices_global = ar_lat_indices_global[::-1]

    seasonal_metrics_data = {
        metric: {} for metric in ["p_ext_ar", "p_ext_no_ar", "paf", "af", "lift"]
    }

    for season_name, months in seasons.items():

        seasonal_mask = np.asarray([date.month in months for date in common_dates])

        seasonal_ar_indices = all_ar_indices[seasonal_mask]
        seasonal_precip_indices = all_precip_indices[seasonal_mask]

        season_output_dir = os.path.join(base_output_dir, "seasonal", season_name)

        metrics_data = run_analysis_parallel(
            seasonal_ar_indices,
            seasonal_precip_indices,
            ar_lat_indices_global,
            ar_lon_indices_global,
            precip_lat_indices_global,
            precip_lon_indices_global,
            precip_lats_global,
            precip_lons_global,
            season_output_dir,
            f"Global_{season_name}",
            lon_min_g,
            lon_max_g,
            lat_min_g,
            lat_max_g,
            date_range_str_for_output
        )

        if metrics_data is not None:
            for metric_key, data in metrics_data.items():
                seasonal_metrics_data[metric_key][season_name] = data

    # ========================================================
    # Combined Seasonal Subplots
    # ========================================================

    if seasonal_metrics_data:
        print("\n--- Generating Combined Seasonal Subplots for all metrics ---")

        combined_output_dir = os.path.join(base_output_dir, "seasonal_combined")

        plot_all_seasonal_subfigures(
            seasonal_metrics_data,
            combined_output_dir,
            lon_min_g,
            lon_max_g,
            lat_min_g,
            lat_max_g,
            precip_lons_global,
            precip_lats_global
        )

        print("Combined seasonal subplots saved.")

    print("\nAll analyses finished.")
