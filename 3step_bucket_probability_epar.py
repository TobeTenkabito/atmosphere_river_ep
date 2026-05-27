import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from netCDF4 import num2date
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import fisher_exact
from scipy.ndimage import uniform_filter
import os
import pandas as pd
from tqdm import tqdm

grid_aggregate_size = 1
sig_level = 0.05
lift_cap_percentile = 95
lift_max_val = 10
output_dir = "G:/ar_analysis/output_probability"
os.makedirs(output_dir, exist_ok=True)
control_modes = ["ENSO", "MJO", "AO", "PDO"]
mode_lags = {"ENSO": 0, "MJO": 0, "AO": 0, "PDO": 0}


def load_enso():
    enso_file = "G:/precipitation/nino3.4.csv"
    df = pd.read_csv(enso_file, header=1, index_col=0)
    df.index = pd.to_datetime(df.index, errors='coerce', format='mixed')
    df = df.dropna(subset=[df.columns[0]])
    values = df.iloc[:, 0].astype(float)
    bucket = pd.Series(index=values.index, dtype="object")
    bucket[values > 0.5] = "ElNino"
    bucket[values < -0.5] = "LaNina"
    bucket[(values >= -0.5) & (values <= 0.5)] = "Neutral"
    return bucket


def load_mjo():
    mjo_file = "G:/precipitation/MJO.csv"
    df = pd.read_csv(mjo_file, header=2)
    df.columns = ['year', 'month', 'day', 'RMM1', 'RMM2', 'RMM_phase', 'RMM_amplitude', 'RMM_weight', "Column9",
                  "Column10"]
    df['time'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    df = df.dropna(subset=['time'])
    df = df.set_index('time')
    amp = df['RMM_amplitude']
    phase = df['RMM_phase']
    bucket = pd.Series(index=df.index, dtype="object")
    bucket[amp < 1] = "Inactive"
    bucket[(amp >= 1) & (phase.isin([1, 2, 3, 4]))] = "Active_1to4"
    bucket[(amp >= 1) & (phase.isin([5, 6, 7, 8]))] = "Active_5to8"
    return bucket


def load_ao():
    ao_file = "G:/precipitation/AO.csv"
    df = pd.read_csv(ao_file)
    df['time'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')
    df = df.set_index('time')
    values = df['index'].astype(float)
    bucket = pd.Series(index=values.index, dtype="object")
    bucket[values > 0.5] = "Positive"
    bucket[values < -0.5] = "Negative"
    bucket[(values >= -0.5) & (values <= 0.5)] = "Neutral"
    return bucket


def load_pdo():
    pdo_file = "G:/precipitation/PDO.csv"
    df = pd.read_csv(pdo_file)
    df_long = pd.melt(df, id_vars=['Year'], var_name='Month', value_name='index')
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                 'Nov': 11, 'Dec': 12}
    df_long['MonthNum'] = df_long['Month'].map(month_map)
    df_long['time'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['MonthNum'].astype(str),
                                     format='%Y-%m')
    df_long = df_long.set_index('time').sort_index()
    values = df_long['index'].astype(float)
    bucket = pd.Series(index=values.index, dtype="object")
    bucket[values > 0.5] = "Positive"
    bucket[values < -0.5] = "Negative"
    bucket[(values >= -0.5) & (values <= 0.5)] = "Neutral"
    return bucket


def shift_series_monthly(series, lag_months):
    """对时间序列做月滞后，正数表示向后平移"""
    if lag_months == 0:
        return series
    # 将 index 转换为月初时间戳，避免日对齐问题
    idx = pd.to_datetime(series.index).to_period("M").to_timestamp()
    series.index = idx
    shifted = series.copy()
    shifted.index = shifted.index + pd.DateOffset(months=lag_months)
    return shifted


def get_bucket_labels(dates_series):
    labels = {}
    if "ENSO" in control_modes:
        enso = load_enso()
        enso = shift_series_monthly(enso, mode_lags.get("ENSO", 0))
        labels["ENSO"] = enso.reindex(dates_series, method='nearest').values
    if "MJO" in control_modes:
        mjo = load_mjo()
        mjo = shift_series_monthly(mjo, mode_lags.get("MJO", 0))
        labels["MJO"] = mjo.reindex(dates_series, method='nearest').values
    if "AO" in control_modes:
        ao = load_ao()
        ao = shift_series_monthly(ao, mode_lags.get("AO", 0))
        labels["AO"] = ao.reindex(dates_series, method='nearest').values
    if "PDO" in control_modes:
        pdo = load_pdo()
        pdo = shift_series_monthly(pdo, mode_lags.get("PDO", 0))
        labels["PDO"] = pdo.reindex(dates_series, method='nearest').values
    return labels


def compute_bucket_stats(ar_data, precip_data, bucket_labels):
    results = {}
    for mode in control_modes:
        buckets = np.unique(bucket_labels[mode])
        results[mode] = {}
        for bucket in buckets:
            mask = bucket_labels[mode] == bucket
            print(f"{mode}={bucket}: {np.sum(mask)} time points")
            if np.sum(mask) == 0:
                continue
            ar_subset = ar_data[mask]
            precip_subset = precip_data[mask]
            A = np.sum((ar_subset == 1) & (precip_subset == 1), axis=0)  # AR and EP
            B = np.sum((ar_subset == 1) & (precip_subset == 0), axis=0)  # AR and no EP
            C = np.sum((ar_subset == 0) & (precip_subset == 1), axis=0)  # no AR and EP
            D = np.sum((ar_subset == 0) & (precip_subset == 0), axis=0)  # no AR and no EP

            # Prevent division by zero
            denom_AR = np.where((A + B) > 0, A + B, np.nan)
            denom_notAR = np.where((C + D) > 0, C + D, np.nan)

            # P(EP|AR) and P(EP|no AR)
            p_ext_ar = A / denom_AR
            p_ext_no_ar = C / denom_notAR

            # PAF and AF calculation
            N = A + B + C + D
            Freq_E = (A + C) / np.where(N > 0, N, np.nan)

            # AF (Attributable Fraction)
            AF = (p_ext_ar - p_ext_no_ar) / p_ext_ar

            # PAF (Population Attributable Fraction)
            p_notAR = (C + D) / np.where(N > 0, N, np.nan)
            PAF = (Freq_E - p_notAR * p_ext_no_ar) / Freq_E

            p_sig = np.full(A.shape, np.nan)
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    table = [[A[i, j], B[i, j]], [C[i, j], D[i, j]]]
                    if np.sum(table) > 0:
                        _, p = fisher_exact(table, alternative="greater")  # 单侧检验：AR 是否增加 EP
                        p_sig[i, j] = p

            if grid_aggregate_size > 1:
                p_ext_ar = uniform_filter(p_ext_ar, size=grid_aggregate_size, mode='nearest')
                p_ext_no_ar = uniform_filter(p_ext_no_ar, size=grid_aggregate_size, mode='nearest')
                p_sig = uniform_filter(p_sig, size=grid_aggregate_size, mode='nearest')
                AF = uniform_filter(AF, size=grid_aggregate_size, mode='nearest')
                PAF = uniform_filter(PAF, size=grid_aggregate_size, mode='nearest')

            results[mode][bucket] = {
                'p_ext_ar': p_ext_ar,
                'p_ext_no_ar': p_ext_no_ar,
                'p_sig': p_sig,
                'AF': AF,
                'PAF': PAF
            }
    return results


def find_max_prob(results, lats, lons):
    max_prob = np.full((len(lats), len(lons)), np.nan)
    max_mode = np.full((len(lats), len(lons)), '', dtype='object')
    max_p_sig = np.full((len(lats), len(lons)), np.nan)
    for mode in results:
        for bucket in results[mode]:
            p_ext_ar = results[mode][bucket]['p_ext_ar']
            p_sig = results[mode][bucket]['p_sig']
            mask = (p_sig < sig_level) & (np.isnan(max_prob) | (p_ext_ar > max_prob))
            max_prob[mask] = p_ext_ar[mask]
            max_mode[mask] = f"{mode}={bucket}"
            max_p_sig[mask] = p_sig[mask]
    print("max_mode unique values:", np.unique(max_mode))
    return max_prob, max_mode, max_p_sig


def plot_map(data, title, filename, vmin, vmax, sig_mask=None, max_mode=None):
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#D3D3D3')
    ax.add_feature(cfeature.OCEAN, facecolor='#E6F0FA')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
    if max_mode is not None:
        valid_mask = (max_mode != '') & (max_mode != None)
        unique_modes = np.unique(max_mode[valid_mask])
        if len(unique_modes) == 0:
            print(f"Warning: No valid modes found for {filename}. Skipping plot.")
            plt.close()
            return
        colors = plt.cm.get_cmap('tab10', len(unique_modes))
        mode_colors = {mode: colors(i) for i, mode in enumerate(unique_modes)}
        cmap = matplotlib.colors.ListedColormap([mode_colors[mode] for mode in unique_modes])
        norm = matplotlib.colors.BoundaryNorm(np.linspace(0, len(unique_modes), len(unique_modes) + 1), cmap.N)
        mode_num = np.full(max_mode.shape, np.nan)
        for i, mode in enumerate(unique_modes):
            mode_num[max_mode == mode] = i
        mode_num = np.ma.masked_where(sig_mask >= sig_level, mode_num)
        mesh_prob = ax.pcolormesh(lons, lats, np.ma.masked_where(sig_mask >= sig_level, data),
                                  cmap='viridis', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        mesh_mode = ax.pcolormesh(lons, lats, mode_num, cmap=cmap, alpha=0.5, transform=ccrs.PlateCarree())
        cbar_prob = plt.colorbar(mesh_prob, ax=ax, label='Probability', shrink=0.8, location='right')
        cbar_mode = plt.colorbar(mesh_mode, ax=ax, label='Controlling Mode', shrink=0.8, location='bottom')
        cbar_mode.set_ticks(np.arange(len(unique_modes)) + 0.5)
        cbar_mode.set_ticklabels(unique_modes)
    else:
        masked_data = np.ma.masked_where(sig_mask >= sig_level, data) if sig_mask is not None else data
        mesh = ax.pcolormesh(lons, lats, masked_data, cmap='viridis', vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree())
        plt.colorbar(mesh, ax=ax, label='Probability', shrink=0.8, location='right')
    plt.title(f"{title}\n{start_date_input} to {end_date_input}", fontsize=14)
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_map_single_colorbar(data_grid, title, outfile, lons, lats, vmin, vmax, sig_mask=None, cmap="coolwarm"):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # 背景要素
    ax.coastlines(resolution="50m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)

    if sig_mask is not None:
        data_grid = np.where(sig_mask < sig_level, data_grid, np.nan)

    mesh = ax.pcolormesh(lons, lats, data_grid, cmap=cmap, shading="auto",
                         transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    cb = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
    cb.set_label(title)

    plt.title(title, fontsize=14)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("Loading NetCDF datasets...")
    ar_ds = nc.Dataset('G:/ar_analysis/ar_happen.nc')
    precip_ds = nc.Dataset('G:/ar_analysis/data/extreme_precipitation.nc')
    ar_time = ar_ds.variables['time'][:]
    ar_time_units = ar_ds.variables['time'].units
    ar_time_calendar = ar_ds.variables['time'].calendar
    ar_dates = num2date(ar_time, units=ar_time_units, calendar=ar_time_calendar)
    precip_time = precip_ds.variables['time'][:]
    precip_time_units = precip_ds.variables['time'].units
    precip_time_calendar = precip_ds.variables['time'].calendar
    precip_dates = num2date(precip_time, units=precip_time_units, calendar=precip_time_calendar)
    start_date_input = input("起始日期（YYYY-MM-DD HH:MM:SS）: ")
    end_date_input = input("结束日期（YYYY-MM-DD HH:MM:SS）: ")
    start_date = datetime.strptime(start_date_input, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(end_date_input, "%Y-%m-%d %H:%M:%S")
    seasonal_analysis = input("是否进行季节性分析？(y/n): ").lower() == 'y'
    season_months = None
    season_name = ""
    if seasonal_analysis:
        months_input = input("请输入季节月份（用逗号分隔，例如 12,1,2 表示DJF）：")
        season_months = [int(m) for m in months_input.split(',')]
        season_name = f"Season: {months_input.replace(',', '-')}"
    ar_dates_pd = pd.to_datetime([d.strftime('%Y-%m-%d %H:%M:%S') for d in tqdm(ar_dates, desc="Converting AR dates")])
    precip_dates_pd = pd.to_datetime(
        [d.strftime('%Y-%m-%d %H:%M:%S') for d in tqdm(precip_dates, desc="Converting Precip dates")])
    df_ar = pd.DataFrame({'ar_idx': np.arange(len(ar_dates_pd))}, index=ar_dates_pd)
    df_precip = pd.DataFrame({'precip_idx': np.arange(len(precip_dates_pd))}, index=precip_dates_pd)
    df_ar_sel = df_ar[start_date:end_date]
    df_precip_sel = df_precip[start_date:end_date]
    merged_df = df_ar_sel.merge(df_precip_sel, left_index=True, right_index=True, how='inner')
    if merged_df.empty:
        print("错误：在指定的时间范围内，两个数据文件没有共同的时间点。")
        exit()
    ar_indices = merged_df['ar_idx'].values
    precip_indices = merged_df['precip_idx'].values
    ar_dates_sel = ar_dates[ar_indices]
    if seasonal_analysis:
        month_mask = np.isin([d.month for d in ar_dates_sel], season_months)
        ar_indices = ar_indices[month_mask]
        precip_indices = precip_indices[month_mask]
        ar_dates_sel = ar_dates_sel[month_mask]
        if len(ar_dates_sel) == 0:
            print(f"错误：在指定的时间范围和季节（{season_name}）内没有数据。")
            exit()
    lon_min, lon_max = 100, 150
    lat_min, lat_max = 10, 60
    grid_res = 0.25
    lons = np.arange(lon_min, lon_max + grid_res, grid_res)
    lats = np.arange(lat_min, lat_max + grid_res, grid_res)
    precip_lats = precip_ds.variables['latitude'][:]
    precip_lons = precip_ds.variables['longitude'][:]
    lat_mask = (precip_lats >= lat_min) & (precip_lats <= lat_max)
    lon_mask = (precip_lons >= lon_min) & (precip_lons <= lon_max)
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    precip_lats = precip_lats[lat_indices]
    precip_lons = precip_lons[lon_indices]
    ar_lats = ar_ds.variables['lat'][:]
    ar_lons = ar_ds.variables['lon'][:]
    ar_lat_mask = (ar_lats >= lat_min) & (ar_lats <= lat_max)
    ar_lon_mask = (ar_lons >= lon_min) & (ar_lons <= lon_max)
    ar_lat_indices = np.where(ar_lat_mask)[0]
    ar_lon_indices = np.where(ar_lon_mask)[0]
    ar_lats = ar_lats[ar_lat_indices]
    ar_lons = ar_lons[ar_lon_indices]
    if ar_lats.size > 1 and ar_lats[0] > ar_lats[-1]:
        ar_lats, ar_lat_indices = ar_lats[::-1], ar_lat_indices[::-1]
    if precip_lats.size > 1 and precip_lats[0] > precip_lats[-1]:
        precip_lats, lat_indices = precip_lats[::-1], lat_indices[::-1]
    ar_data = ar_ds.variables['ar_happen'][ar_indices, ar_lat_indices, ar_lon_indices].astype(np.int8)
    precip_data = precip_ds.variables['extreme_precipitation_flag'][precip_indices, lat_indices, lon_indices].astype(
        np.int8)
    ar_dates_sel_dt = pd.to_datetime([d.strftime('%Y-%m-%d %H:%M:%S') for d in ar_dates_sel])
    bucket_labels = get_bucket_labels(ar_dates_sel_dt)
    results = compute_bucket_stats(ar_data, precip_data, bucket_labels)
    all_sig_paf = []
    all_sig_af = []

    for mode in results:
        for bucket in results[mode]:
            p_ext_ar = results[mode][bucket]['p_ext_ar']
            p_ext_no_ar = results[mode][bucket]['p_ext_no_ar']
            p_sig = results[mode][bucket]['p_sig']
            AF = results[mode][bucket]['AF']
            PAF = results[mode][bucket]['PAF']

            plot_map(p_ext_ar, f'P(EP|AR, {mode}={bucket})', f'p_ext_ar_{mode}_{bucket}', vmin=0, vmax=1,
                     sig_mask=p_sig)
            plot_map(p_ext_no_ar, f'P(EP|no AR, {mode}={bucket})', f'p_ext_no_ar_{mode}_{bucket}', vmin=0, vmax=1,
                     sig_mask=p_sig)

            # 绘制 AF 和 PAF 地图
            plot_map_single_colorbar(AF, f'AF ({mode}={bucket})', f'AF_map_{mode}_{bucket}.png',
                                     lons, lats, vmin=0, vmax=1, sig_mask=p_sig, cmap="coolwarm")
            plot_map_single_colorbar(PAF, f'PAF ({mode}={bucket})', f'PAF_map_{mode}_{bucket}.png',
                                     lons, lats, vmin=0, vmax=1, sig_mask=p_sig, cmap="coolwarm")

            # 收集显著格点的 AF 和 PAF
            sig_paf = PAF[p_sig < sig_level]
            sig_af = AF[p_sig < sig_level]
            all_sig_paf.extend(sig_paf[~np.isnan(sig_paf)])
            all_sig_af.extend(sig_af[~np.isnan(sig_af)])

    max_prob, max_mode, max_p_sig = find_max_prob(results, ar_lats, ar_lons)
    plot_map(max_prob, 'Maximum P(EP|AR) by Controlling Mode', 'max_p_ext_ar', vmin=0, vmax=1, sig_mask=max_p_sig,
             max_mode=max_mode)

    mean_paf = np.nanmean(all_sig_paf) if all_sig_paf else np.nan
    mean_af = np.nanmean(all_sig_af) if all_sig_af else np.nan

    print(f"Global mean PAF (p<0.05): {mean_paf:.4f}")
    print(f"Global mean AF (p<0.05): {mean_af:.4f}")

    with open(f"{output_dir}/summary_stats.txt", "w") as f:
        f.write("=== Mode lag settings (months) ===\n")
        for mode, lag in mode_lags.items():
            f.write(f"{mode}: lag = {lag} months\n")
        f.write("\n=== Statistics ===\n")

        for mode in results:
            for bucket in results[mode]:
                p_ext_ar = results[mode][bucket]['p_ext_ar']
                p_ext_no_ar = results[mode][bucket]['p_ext_no_ar']
                p_sig = results[mode][bucket]['p_sig']
                AF = results[mode][bucket]['AF']
                PAF = results[mode][bucket]['PAF']

                f.write(f"Mean P(EP|AR, {mode}={bucket}): {np.nanmean(p_ext_ar):.3f}\n")
                f.write(f"Mean P(EP|no AR, {mode}={bucket}): {np.nanmean(p_ext_no_ar):.3f}\n")
                f.write(f"Mean AF ({mode}={bucket}, p<0.05): {np.nanmean(AF[p_sig < sig_level]):.3f}\n")
                f.write(f"Mean PAF ({mode}={bucket}, p<0.05): {np.nanmean(PAF[p_sig < sig_level]):.3f}\n")
                f.write(f"Significant grid points (p < {sig_level}, {mode}={bucket}): {np.sum(p_sig < sig_level)}\n")
        f.write(f"Effective grid points (Max P): {np.sum(~np.isnan(max_prob))}\n")
        f.write("\n=== Global Averages (for all significant points) ===\n")
        f.write(f"Global mean PAF (p<0.05): {mean_paf:.4f}\n")
        f.write(f"Global mean AF (p<0.05): {mean_af:.4f}\n")

    print("绘图与统计完成，输出文件已保存到:", output_dir)
    ar_ds.close()
    precip_ds.close()
