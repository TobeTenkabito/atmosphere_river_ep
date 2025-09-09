import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from netCDF4 import num2date
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import pandas as pd
from tqdm import tqdm
from scipy.stats import pointbiserialr

# ----------------- 参数 -----------------
grid_aggregate_size = 1
sig_level = 0.05
output_dir = "G:/ar_analysis/output_final"
os.makedirs(output_dir, exist_ok=True)
control_modes = ["ENSO", "MJO", "AO", "PDO"]

# 样本阈值
N_min = 30
n_AR_min = 10
fdr_alpha = 0.05

# 季节性分析参数
seasonal_analysis = input("是否进行季节性分析？（输入'YES'或'None'）：")  # 用户输入'YES'或'None'

# 季节定义（北半球）
seasons = {
    'MAM': [3, 4, 5],  # 春季：3月、4月、5月
    'JJA': [6, 7, 8],  # 夏季：6月、7月、8月
    'SON': [9, 10, 11],  # 秋季：9月、10月、11月
    'DJF': [12, 1, 2]  # 冬季：12月、1月、2月
}


# ----------------- 指数加载函数 -----------------
def load_enso():
    enso_file = "G:/precipitation/nino3.4.csv"
    df = pd.read_csv(enso_file, header=1, index_col=0)
    df.index = pd.to_datetime(df.index, errors='coerce', format='mixed')
    df = df.dropna(subset=[df.columns[0]])
    values = df.iloc[:, 0].astype(float)
    return pd.Series(values, index=df.index)


def load_mjo():
    mjo_file = "G:/precipitation/MJO.csv"
    df = pd.read_csv(mjo_file, header=2)
    df.columns = ['year', 'month', 'day', 'RMM1', 'RMM2', 'RMM_phase',
                  'RMM_amplitude', 'RMM_weight', "Column9", "Column10"]
    df['time'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    df = df.dropna(subset=['time'])
    df = df.set_index('time')
    return df['RMM_amplitude']


def load_ao():
    ao_file = "G:/precipitation/AO.csv"
    df = pd.read_csv(ao_file)
    df['time'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str),
                                format='%Y-%m')
    df = df.set_index('time')
    values = df['index'].astype(float)
    return pd.Series(values, index=df.index)


def load_pdo():
    pdo_file = "G:/precipitation/PDO.csv"
    df = pd.read_csv(pdo_file)
    df_long = pd.melt(df, id_vars=['Year'], var_name='Month', value_name='index')
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    df_long['MonthNum'] = df_long['Month'].map(month_map)
    df_long['time'] = pd.to_datetime(df_long['Year'].astype(str) + '-' +
                                     df_long['MonthNum'].astype(str), format='%Y-%m')
    df_long = df_long.set_index('time').sort_index()
    values = df_long['index'].astype(float)
    return pd.Series(values, index=df_long.index)


def get_mode_values(dates_series):
    values = {}
    if "ENSO" in control_modes:
        values["ENSO"] = load_enso().reindex(dates_series, method='nearest').values
    if "MJO" in control_modes:
        values["MJO"] = load_mjo().reindex(dates_series, method='nearest').values
    if "AO" in control_modes:
        values["AO"] = load_ao().reindex(dates_series, method='nearest').values
    if "PDO" in control_modes:
        values["PDO"] = load_pdo().reindex(dates_series, method='nearest').values
    return values


# ----------------- 核心计算 -----------------
def compute_correlation_stats(ar_data, precip_data, mode_values, time_indices=None):
    if time_indices is None:
        time_indices = np.arange(ar_data.shape[0])
    ar_data = ar_data[time_indices]
    precip_data = precip_data[time_indices]
    mode_values = {mode: values[time_indices] for mode, values in mode_values.items()}
    ar_and_ep = (ar_data == 1) & (precip_data == 1).astype(np.int8)
    results = {}
    for mode in control_modes:
        mode_series = mode_values[mode]
        corr = np.full(ar_data.shape[1:], np.nan)
        p_wald = np.full(ar_data.shape[1:], np.nan)
        N = np.sum(~np.isnan(ar_and_ep), axis=0)
        ARcount = np.sum(ar_and_ep, axis=0)
        for y in tqdm(range(ar_data.shape[1]), desc=f"Computing {mode} correlation"):
            for x in range(ar_data.shape[2]):
                if N[y, x] < N_min or ARcount[y, x] < n_AR_min:
                    continue
                ar_ep_series = ar_and_ep[:, y, x]
                valid = ~np.isnan(ar_ep_series) & ~np.isnan(mode_series)
                if np.sum(valid) < N_min:
                    continue
                r, p = pointbiserialr(ar_ep_series[valid], mode_series[valid])
                corr[y, x] = r
                p_wald[y, x] = p
        results[mode] = {
            'corr': corr,
            'p_wald': p_wald,
            'N': N,
            'ARcount': ARcount
        }
    return results


def bh_fdr_mask(p_array, alpha=0.05):
    p_flat = p_array.flatten()
    valid = ~np.isnan(p_flat)
    pv = p_flat[valid]
    n = len(pv)
    if n == 0:
        return np.zeros_like(p_array, dtype=bool)
    order = np.argsort(pv)
    pv_sorted = pv[order]
    ranks = np.arange(1, n + 1)
    crit = (ranks / n) * alpha
    passed = pv_sorted <= crit
    if not np.any(passed):
        return np.zeros_like(p_array, dtype=bool)
    k = np.max(np.where(passed)[0])
    p_thresh = pv_sorted[k]
    mask = np.zeros_like(p_flat, dtype=bool)
    mask[valid] = p_flat[valid] <= p_thresh
    return mask.reshape(p_array.shape)


# ----------------- 绘图函数 -----------------
def plot_map(data, lons, lats, title, filename, cmap='RdBu_r', vmin=None, vmax=None, cbar_label='Value'):
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    mesh = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    plt.colorbar(mesh, ax=ax, label=cbar_label, shrink=0.8)
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_dominance_map(dominant_mode, lons, lats, title, filename):
    unique_modes = np.unique(dominant_mode)
    unique_modes = [m for m in unique_modes if m != '' and m != 'no-dominant']
    n = len(unique_modes)
    cmap = plt.get_cmap('tab20', max(2, n))
    mode2idx = {mode: i for i, mode in enumerate(unique_modes)}
    mode_num = np.full(dominant_mode.shape, np.nan)
    for mode, idx in mode2idx.items():
        mode_num[dominant_mode == mode] = idx
    mask = (dominant_mode == 'no-dominant')
    mode_num_masked = np.ma.masked_where(mask, mode_num)

    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    mesh = ax.pcolormesh(lons, lats, mode_num_masked, cmap=cmap, transform=ccrs.PlateCarree())

    cbar = plt.colorbar(mesh, ax=ax, ticks=np.arange(n), orientation='horizontal', pad=0.05)
    cbar.ax.set_xticklabels(unique_modes, rotation=0, ha='center', va='top')

    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


# ----------------- 区域汇总 -----------------
def region_summary(results, ar_lats, ar_lons, season=None):
    regions = {
        'SouthChina': (18, 24, 108, 116),
        'NorthChina': (34, 41, 112, 122),
        'NEAsia': (40, 52, 122, 142),
        'JapanKorea': (30, 42, 128, 144)
    }
    rows = []
    for region_name, (lat1, lat2, lon1, lon2) in regions.items():
        lat_mask_reg = (ar_lats >= lat1) & (ar_lats <= lat2)
        lon_mask_reg = (ar_lons >= lon1) & (ar_lons <= lon2)
        for mode in results:
            corr = results[mode]['corr'][np.ix_(lat_mask_reg, lon_mask_reg)]
            mean_corr = np.nanmean(corr)
            rows.append([region_name, mode, mean_corr, season if season else 'Annual'])
    df_region = pd.DataFrame(rows, columns=['region', 'mode', 'mean_corr', 'season'])
    filename = f'region_summary_{season}.csv' if season else 'region_summary.csv'
    df_region.to_csv(os.path.join(output_dir, filename), index=False)


# ----------------- 主程序 -----------------
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

    start_date_input = input("起始日期（YYYY-MM-DD HH:MM:SS）：")
    end_date_input = input("结束日期（YYYY-MM-DD HH:MM:SS）：")
    start_date = datetime.strptime(start_date_input, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(end_date_input, "%Y-%m-%d %H:%M:%S")

    ar_dates_pd = pd.to_datetime([d.strftime('%Y-%m-%d %H:%M:%S') for d in tqdm(ar_dates, desc="Converting AR dates")])
    precip_dates_pd = pd.to_datetime(
        [d.strftime('%Y-%m-%d %H:%M:%S') for d in tqdm(precip_dates, desc="Converting Precip dates")])

    df_ar = pd.DataFrame({'ar_idx': np.arange(len(ar_dates_pd))}, index=ar_dates_pd)
    df_precip = pd.DataFrame({'precip_idx': np.arange(len(precip_dates_pd))}, index=precip_dates_pd)
    df_ar_sel = df_ar[start_date:end_date]
    df_precip_sel = df_precip[start_date:end_date]
    merged_df = df_ar_sel.merge(df_precip_sel, left_index=True, right_index=True, how='inner')
    if merged_df.empty:
        print("错误：在指定的时间范围内没有共同时间点。")
        exit()

    ar_indices = merged_df['ar_idx'].values
    precip_indices = merged_df['precip_idx'].values
    ar_dates_sel_dt = pd.to_datetime([d.strftime('%Y-%m-%d %H:%M:%S') for d in ar_dates[ar_indices]])

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
    if ar_lats[0] > ar_lats[-1]:
        ar_lats = ar_lats[::-1];
        ar_lat_indices = ar_lat_indices[::-1]
    if precip_lats[0] > precip_lats[-1]:
        precip_lats = precip_lats[::-1];
        lat_indices = lat_indices[::-1]

    ar_data = ar_ds.variables['ar_happen'][ar_indices, ar_lat_indices, ar_lon_indices].astype(np.int8)
    precip_data = precip_ds.variables['extreme_precipitation_flag'][precip_indices, lat_indices, lon_indices].astype(
        np.int8)

    mode_values = get_mode_values(ar_dates_sel_dt)

    # 季节性分析
    if seasonal_analysis == 'YES':
        for season, months in seasons.items():
            print(f"Processing season: {season}")
            # 提取季节性索引
            season_mask = np.isin([d.month for d in ar_dates_sel_dt], months)
            season_indices = np.where(season_mask)[0]
            if len(season_indices) < N_min:
                print(f"警告：{season} 的样本量（{len(season_indices)}）小于{N_min}，跳过该季节")
                continue

            # 计算季节性相关性
            results = compute_correlation_stats(ar_data, precip_data, mode_values, season_indices)

            # 构建堆栈
            M = len(control_modes)
            Y, X = len(ar_lats), len(ar_lons)
            corr_stack = np.full((M, Y, X), np.nan)
            p_stack = np.full((M, Y, X), np.nan)
            N_stack = np.full((M, Y, X), 0, dtype=int)
            for m, mode in enumerate(control_modes):
                corr_stack[m, :, :] = results[mode]['corr']
                p_stack[m, :, :] = results[mode]['p_wald']
                N_stack[m, :, :] = results[mode]['N']

            # 应用FDR和样本筛选
            corr_masked = np.full(corr_stack.shape, np.nan)
            for m in range(M):
                valid = (N_stack[m, :, :] >= N_min) & (results[control_modes[m]]['ARcount'] >= n_AR_min)
                mask_fdr = bh_fdr_mask(p_stack[m, :, :], fdr_alpha)
                final_mask = valid & mask_fdr
                corr_masked[m, :, :][final_mask] = corr_stack[m, :, :][final_mask]

            # 确定主要控制模态
            nan_slices = np.all(np.isnan(corr_masked), axis=0)
            if np.any(nan_slices):
                print(f"{season} 找到全 NaN 切片，位置：", np.where(nan_slices))
                max_idx = np.full((Y, X), -1, dtype=int)
                max_val = np.full((Y, X), np.nan)
                for y in range(Y):
                    for x in range(X):
                        if not nan_slices[y, x]:
                            max_idx[y, x] = np.nanargmax(np.abs(corr_masked[:, y, x]))
                            max_val[y, x] = np.nanmax(np.abs(corr_masked[:, y, x]))
            else:
                max_idx = np.nanargmax(np.abs(corr_masked), axis=0)
                max_val = np.nanmax(np.abs(corr_masked), axis=0)
            dominant_mode = np.full((Y, X), 'no-dominant', dtype=object)
            for y in range(Y):
                for x in range(X):
                    if np.isnan(max_val[y, x]) or max_val[y, x] == 0:
                        continue
                    m = max_idx[y, x]
                    dominant_mode[y, x] = control_modes[m]

            # 绘制季节性地图
            for m, mode in enumerate(control_modes):
                plot_map(corr_masked[m, :, :], ar_lons, ar_lats,
                         title=f"{mode} Correlation ({season})",
                         filename=f"{mode}_corr_{season}.png",
                         cmap='RdBu_r', vmin=-0.5, vmax=0.5, cbar_label='Point-Biserial Correlation')
                plot_map(p_stack[m, :, :], ar_lons, ar_lats,
                         title=f"{mode} p-value ({season})",
                         filename=f"{mode}_p_{season}.png",
                         cmap='viridis', vmin=0, vmax=0.1, cbar_label='p-value')
                plot_map(N_stack[m, :, :], ar_lons, ar_lats,
                         title=f"{mode} sample size ({season})",
                         filename=f"{mode}_N_{season}.png",
                         cmap='viridis', cbar_label='N')

            plot_dominance_map(dominant_mode, ar_lons, ar_lats,
                               title=f"Dominant Mode ({season}, {start_date} to {end_date})",
                               filename=f"dominance_map_{season}.png")

            # 区域统计
            region_summary(results, ar_lats, ar_lons, season=season)

    # 全年分析（默认）
    if seasonal_analysis == 'None' or seasonal_analysis == 'YES':
        print("Processing annual analysis...")
        results = compute_correlation_stats(ar_data, precip_data, mode_values)

        # 构建堆栈
        M = len(control_modes)
        Y, X = len(ar_lats), len(ar_lons)
        corr_stack = np.full((M, Y, X), np.nan)
        p_stack = np.full((M, Y, X), np.nan)
        N_stack = np.full((M, Y, X), 0, dtype=int)
        for m, mode in enumerate(control_modes):
            corr_stack[m, :, :] = results[mode]['corr']
            p_stack[m, :, :] = results[mode]['p_wald']
            N_stack[m, :, :] = results[mode]['N']

        # 应用FDR和样本筛选
        corr_masked = np.full(corr_stack.shape, np.nan)
        for m in range(M):
            valid = (N_stack[m, :, :] >= N_min) & (results[control_modes[m]]['ARcount'] >= n_AR_min)
            mask_fdr = bh_fdr_mask(p_stack[m, :, :], fdr_alpha)
            final_mask = valid & mask_fdr
            corr_masked[m, :, :][final_mask] = corr_stack[m, :, :][final_mask]

        # 确定主要控制模态
        nan_slices = np.all(np.isnan(corr_masked), axis=0)
        if np.any(nan_slices):
            print("Annual 找到全 NaN 切片，位置：", np.where(nan_slices))
            max_idx = np.full((Y, X), -1, dtype=int)
            max_val = np.full((Y, X), np.nan)
            for y in range(Y):
                for x in range(X):
                    if not nan_slices[y, x]:
                        max_idx[y, x] = np.nanargmax(np.abs(corr_masked[:, y, x]))
                        max_val[y, x] = np.nanmax(np.abs(corr_masked[:, y, x]))
        else:
            max_idx = np.nanargmax(np.abs(corr_masked), axis=0)
            max_val = np.nanmax(np.abs(corr_masked), axis=0)
        dominant_mode = np.full((Y, X), 'no-dominant', dtype=object)
        for y in range(Y):
            for x in range(X):
                if np.isnan(max_val[y, x]) or max_val[y, x] == 0:
                    continue
                m = max_idx[y, x]
                dominant_mode[y, x] = control_modes[m]

        # 绘制全年地图
        for m, mode in enumerate(control_modes):
            plot_map(corr_masked[m, :, :], ar_lons, ar_lats,
                     title=f"{mode} Correlation (Annual)",
                     filename=f"{mode}_corr.png",
                     cmap='RdBu_r', vmin=-0.5, vmax=0.5, cbar_label='Point-Biserial Correlation')
            plot_map(p_stack[m, :, :], ar_lons, ar_lats,
                     title=f"{mode} p-value (Annual)",
                     filename=f"{mode}_p.png",
                     cmap='viridis', vmin=0, vmax=0.1, cbar_label='p-value')
            plot_map(N_stack[m, :, :], ar_lons, ar_lats,
                     title=f"{mode} sample size (Annual)",
                     filename=f"{mode}_N.png",
                     cmap='viridis', cbar_label='N')

        plot_dominance_map(dominant_mode, ar_lons, ar_lats,
                           title=f"Dominant Mode (Annual, {start_date} to {end_date})",
                           filename=f"dominance_map.png")

        # 区域统计
        region_summary(results, ar_lats, ar_lons)

    print("全部完成。输出在", output_dir)