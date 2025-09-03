"""
Diagnostics module for selecting lag L and determining the number of monthly shuffle iterations.
Functions:
1) Calculates and plots the Event-Triggered Average (ETA): E[A_{t-tau} | E_t = 1], used to identify the lead-lag window of AR versus EP;
2) Calculates and plots the binary cross-correlation (via FFT) and a directionality indicator, S(L) = (r(+L)-r(-L))/max(...);
3) Calculates the "hit-CDF": For each L, calculates the percentage of EPs that hit AR within ±L days, and calculates the CDF of the hit rate versus L;
4) Implements adaptive stopping for sequential Monte Carlo (within-month permutation): plots p_hat(n) against the 95% CI in real time, automatically stopping at a specified error threshold or maximum iterations;
5) Saves all images to output_dir and returns numerical results for further decision-making.

Note:
- Input time series should be aligned (same length, same time step, daily resolution recommended).
- ar_series and ep_series are expected to be binary (0/1). Non-binary region aggregations are also supported (the module automatically binarizes/thresholds the data).
- The monthly shuffle implementation shuffles the date array by month (randomly permuting the AR series within each month).
"""
"""
Diagnostics module for selecting lag L and determining number of monthly-shuffle iterations.
功能：
1) 计算并绘出 Event-Triggered Average (ETA)：E[A_{t-tau} | E_t = 1] 的曲线，用以发现 AR 对 EP 的领先滞后窗口；
2) 计算并绘出二值互相关（通过 FFT）以及一个方向性指标 S(L) = (r(+L)-r(-L))/max(...)；
3) 计算 "hit-CDF"：对于每个 L，统计多少比例的 EP 在 ±L 天内命中 AR，得到命中率随 L 的 CDF；
4) 顺序蒙特卡洛（月内置换）置换的自适应停止实现：实时绘制 p_hat(n) 与 95% CI，自动停止于指定误差阈值或最大迭代；
5) 所有图像保存到 output_dir，并返回数值结果供进一步决策。

注意：
- 输入时间序列应一致对齐（相同长度、相同时间步长，建议为日分辨率）。
- ar_series 和 ep_series 预期为二值（0/1）。也支持非二值的区域聚合（模块中会自动二值化/阈值化）。
- monthly shuffle 的实现会依据日期数组的月份分层打乱（在每个月内部随机置换 AR 序列）。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
import os
from numpy.fft import fft, ifft
from math import sqrt
import xarray as xr
import warnings
from joblib import Parallel, delayed
def ensure_binary(arr, threshold=0.5):
    arr = np.asarray(arr)
    if arr.dtype == np.bool_ or set(np.unique(arr)).issubset({0, 1}):
        return arr.astype(np.int8)
    else:
        return (arr > threshold).astype(np.int8)
def dates_to_month_indices(dates):
    dates = pd.to_datetime(dates)
    return dates.year * 12 + dates.month
def event_triggered_average(ar_series, ep_series, max_lag=30, aggregate_mode='mean', region_mask=None):
    ar = ensure_binary(ar_series)
    ep = ensure_binary(ep_series)
    n = len(ar)
    if not len(ep) == n:
        warnings.warn("ar_series and ep_series must be same length", UserWarning)
        return np.arange(-max_lag, max_lag + 1), np.full(2 * max_lag + 1, np.nan), np.nan
    taus = np.arange(-max_lag, max_lag + 1)
    eta = np.full(len(taus), np.nan, dtype=float)
    ep_idx = np.where(ep == 1)[0]
    if ep_idx.size == 0:
        return taus, eta, np.nan
    for k, tau in enumerate(taus):
        shifted_indices = ep_idx - tau  # because ETA(tau) = ar_{t-tau} given ep_t=1
        valid = (shifted_indices >= 0) & (shifted_indices < n)
        if valid.sum() == 0:
            eta[k] = np.nan
        else:
            eta[k] = np.mean(ar[shifted_indices[valid]])
    baseline = np.mean(ar)
    return taus, eta, baseline
def binary_cross_correlation(ar_series, ep_series, max_lag=30):
    x = np.asarray(ar_series, dtype=float) - np.mean(ar_series)
    y = np.asarray(ep_series, dtype=float) - np.mean(ep_series)
    n = len(x)
    nfft = 1 << (int(np.ceil(np.log2(2 * n - 1))))
    fx = fft(x, nfft)
    fy = fft(y, nfft)
    cc = np.real(ifft(fx * np.conjugate(fy)))
    cc = np.concatenate((cc[-(n - 1):], cc[:n]))
    denom = np.std(x) * np.std(y) * n
    if denom == 0:
        r = np.zeros(2 * max_lag + 1)
    else:
        mid = n - 1
        lags = np.arange(-max_lag, max_lag + 1)
        r = cc[mid + lags] / denom
    taus = np.arange(-max_lag, max_lag + 1)
    return taus, r


def directionality_metric(r_vals):
    Lmax = (len(r_vals) - 1) // 2
    mid = Lmax
    S = np.zeros(Lmax + 1)
    for L in range(1, Lmax + 1):
        r_pos = r_vals[mid + L]
        r_neg = r_vals[mid - L]
        denom = max(abs(r_pos), abs(r_neg), 1e-12)
        S[L] = (r_pos - r_neg) / denom
    return S
def hit_cdf(ar_series, ep_series, max_L=30):
    ar = ensure_binary(ar_series)
    ep = ensure_binary(ep_series)
    ep_idx = np.where(ep == 1)[0]
    n_ep = ep_idx.size
    if n_ep == 0:
        return np.arange(max_L + 1), np.zeros(max_L + 1)
    fractions = np.zeros(max_L + 1, dtype=float)
    n = len(ar)
    for L in range(0, max_L + 1):
        count = 0
        for t in ep_idx:
            lo = max(0, t - L)
            hi = min(n - 1, t + L)
            if np.any(ar[lo:hi + 1] == 1):
                count += 1
        fractions[L] = count / n_ep
    return np.arange(max_L + 1), fractions
def monthly_shuffle_once(arr, months_ids, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    arr = np.array(arr)
    months_ids = np.array(months_ids)
    out = arr.copy()
    for m in np.unique(months_ids):
        idx = np.where(months_ids == m)[0]
        if idx.size <= 1:
            continue
        perm = rng.permutation(idx)
        out[idx] = arr[perm]
    return out


def sequential_monthly_shuffle_test(ar_series, ep_series, dates, statistic_fn,
                                    max_iter=1000, min_iter=10,
                                    abs_err=0.01, rel_err=0.1,
                                    seed=42, verbose=False):
    rng = np.random.default_rng(seed)
    months = dates_to_month_indices(dates)
    if np.sum(ar_series) == 0 or np.sum(ep_series) == 0:
        return np.nan, 0, False
    real_stat = statistic_fn(ar_series, ep_series)
    b = 0
    converged = False
    for n in range(1, max_iter + 1):
        ar_shuf = monthly_shuffle_once(ar_series, months, rng)
        stat = statistic_fn(ar_shuf, ep_series)
        if stat >= real_stat:
            b += 1
        p_hat = (b + 1) / (n + 1)
        se = sqrt(p_hat * (1 - p_hat) / (n + 1))
        halfwidth = 1.96 * se
        if n >= min_iter:
            if halfwidth <= abs_err:
                converged = True
                break
            if p_hat > 0 and (halfwidth / p_hat) <= rel_err:
                converged = True
                break
    else:
        n = max_iter

    p_hat = (b + 1) / (n + 1)
    return p_hat, n, converged
def pooled_OR_statistic(ar_series, ep_series):
    ar = ensure_binary(ar_series)
    ep = ensure_binary(ep_series)
    A = np.sum((ar == 1) & (ep == 1))
    B = np.sum((ar == 1) & (ep == 0))
    C = np.sum((ar == 0) & (ep == 1))
    D = np.sum((ar == 0) & (ep == 0))
    if B * C == 0:
        # fallback to add 0.5 continuity correction
        OR = (A + 0.5) * (D + 0.5) / ((B + 0.5) * (C + 0.5))
    else:
        OR = (A * D) / (B * C)
    return float(OR)
def _calculate_point_diagnostics(ar_series, ep_series, dates):
    try:
        if np.std(ar_series) == 0 or np.std(ep_series) == 0:
            warnings.warn("One or both time series are constant (e.g., all 0s or all 1s). Diagnostics cannot be computed.", UserWarning)
            return (np.nan, np.nan, np.nan)
        if np.sum(ensure_binary(ar_series)) == 0 or np.sum(ensure_binary(ep_series)) == 0:
            return (np.nan, np.nan, np.nan)
        taus, rvals = binary_cross_correlation(ar_series, ep_series, max_lag=15)
        max_r_idx = np.nanargmax(np.abs(rvals))
        max_corr = rvals[max_r_idx]
        max_corr_lag = taus[max_r_idx]
        p_hat, _, _ = sequential_monthly_shuffle_test(
            ar_series, ep_series, dates, pooled_OR_statistic,
            max_iter=500, min_iter=10, verbose=False
        )
        return (max_corr, max_corr_lag, p_hat)
    except Exception as e:
        warnings.warn(f"An error occurred during diagnostics for this grid point. Returning NaNs. Error: {e}", UserWarning)
        return (np.nan, np.nan, np.nan)
def run_spatial_diagnostics(ar_ds, ep_ds, output_dir="diagnostics_output", n_jobs=None):
    os.makedirs(output_dir, exist_ok=True)
    dates = ar_ds['time'].values

    print("[INFO] Extracting data into memory (per spatial point). This may take a while...")
    ar_data = ar_ds['ar_happen'].compute().values  # shape: (time, lat, lon)
    ep_data = ep_ds['extreme_precipitation_flag'].compute().values

    nt, ny, nx = ar_data.shape
    results = np.full((3, ny, nx), np.nan, dtype=float)

    print(f"[INFO] Running diagnostics in parallel on {ny*nx} grid points...")
    def process_point(j, i):
        ar_series = ar_data[:, j, i]
        ep_series = ep_data[:, j, i]
        return _calculate_point_diagnostics(ar_series, ep_series, dates)

    # 使用 joblib 并行
    out = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(process_point)(j, i)
        for j in range(ny) for i in range(nx)
    )

    # 还原结果
    for idx, (mc, lag, p) in enumerate(out):
        j, i = divmod(idx, nx)
        results[0, j, i] = mc
        results[1, j, i] = lag
        results[2, j, i] = p

    # 转为 xarray
    max_corr = xr.DataArray(results[0], coords=[ar_ds.lat, ar_ds.lon], dims=['lat', 'lon'], name="max_cross_correlation")
    max_corr_lag = xr.DataArray(results[1], coords=[ar_ds.lat, ar_ds.lon], dims=['lat', 'lon'], name="max_cross_correlation_lag")
    p_values = xr.DataArray(results[2], coords=[ar_ds.lat, ar_ds.lon], dims=['lat', 'lon'], name="p_value")

    print("[INFO] Diagnostics complete. Generating maps...")
    plt.figure(figsize=(10, 7))
    max_corr.plot(cmap='RdBu_r', norm=colors.TwoSlopeNorm(vcenter=0), cbar_kwargs={'label': 'max correlation (r)'})
    plt.title('Spatial distribution map: maximum correlation between AR and EP')
    plt.savefig(os.path.join(output_dir, "spatial_max_correlation.png"))
    plt.close()

    plt.figure(figsize=(10, 7))
    max_corr_lag.plot(cmap='Spectral_r', cbar_kwargs={'label': '滞后天数'})
    plt.title('Spatial distribution map: optimal lag time for AR and EP')
    plt.savefig(os.path.join(output_dir, "spatial_max_lag.png"))
    plt.close()

    plt.figure(figsize=(10, 7))
    p_values.plot(cmap='viridis_r', cbar_kwargs={'label': 'p-值'})
    plt.title('Spatial distribution map: p-value of the association between AR and EP')
    plt.savefig(os.path.join(output_dir, "spatial_p_value.png"))
    plt.close()

    # --- CSV 文件导出功能 ---
    print("[INFO] Exporting numerical results to CSV...")
    # 将xarray数据转换为DataFrame
    df_max_corr = max_corr.to_dataframe()
    df_max_corr_lag = max_corr_lag.to_dataframe()
    df_p_values = p_values.to_dataframe()

    # 合并所有数据
    df = df_max_corr.join([df_max_corr_lag, df_p_values]).reset_index()

    # 保存为CSV文件
    csv_path = os.path.join(output_dir, "diagnostics_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Numerical results saved to {csv_path}")

    print(f"所有诊断结果的地理分布图已保存到 {output_dir} 文件夹中。")
    return max_corr, max_corr_lag, p_values

if __name__ == "__main__":
    ar_file = "G:/ar_analysis/ar_happen.nc"
    ep_file = "G:/ar_analysis/data/extreme_precipitation.nc"
    output_dir = "diagnostics_output"
    try:
        if not (os.path.exists(ar_file) and os.path.exists(ep_file)):
            print(f"[ERROR] Cannot find input files. Please check paths:\n- {ar_file}\n- {ep_file}")
            raise SystemExit
        print(f"Loading AR data from {ar_file} with Dask...")
        ar_ds = xr.open_dataset(ar_file, chunks={'time': 100, 'lat': 50, 'lon': 50})
        rename_ar = {}
        if 'latitude' in ar_ds.dims:
            rename_ar['latitude'] = 'lat'
        if 'longitude' in ar_ds.dims:
            rename_ar['longitude'] = 'lon'
        if rename_ar:
            ar_ds = ar_ds.rename(rename_ar)
        if 'lat' in ar_ds.coords and ar_ds['lat'].values[0] > ar_ds['lat'].values[-1]:
            print("[INFO] Reversing AR dataset latitude dimension.")
            ar_ds = ar_ds.isel(lat=slice(None, None, -1))
        print(f"Loading EP data from {ep_file} with Dask...")
        ep_ds = xr.open_dataset(ep_file, chunks={'time': 100, 'lat': 50, 'lon': 50})
        rename_ep = {}
        if 'latitude' in ep_ds.dims:
            rename_ep['latitude'] = 'lat'
        if 'longitude' in ep_ds.dims:
            rename_ep['longitude'] = 'lon'
        if rename_ep:
            ep_ds = ep_ds.rename(rename_ep)
        if 'lat' in ep_ds.coords and ep_ds['lat'].values[0] > ep_ds['lat'].values[-1]:
            print("[INFO] Reversing EP dataset latitude dimension.")
            ep_ds = ep_ds.isel(lat=slice(None, None, -1))
        print("[INFO] Resampling data to daily frequency to ensure unique time index...")
        ar_ds_resampled = ar_ds.resample(time='D').max()
        ep_ds_resampled = ep_ds.resample(time='D').max()
        start_date_input = input("Enter start date (YYYY-MM-DD HH:MM:SS): ").strip()
        end_date_input = input("Enter end date (YYYY-MM-DD HH:MM:SS): ").strip()
        start_date_ts = np.datetime64(datetime.strptime(start_date_input, "%Y-%m-%d %H:%M:%S"))
        end_date_ts = np.datetime64(datetime.strptime(end_date_input, "%Y-%m-%d %H:%M:%S"))
        ar_ds_sel = ar_ds_resampled.sel(time=slice(start_date_ts, end_date_ts))
        ep_ds_sel = ep_ds_resampled.sel(time=slice(start_date_ts, end_date_ts))
        if ar_ds_sel.time.size == 0 or ep_ds_sel.time.size == 0:
            print("[ERROR] No common time steps found between the datasets for the selected period.")
            raise SystemExit
        ar_ds_sel, ep_ds_sel = xr.align(ar_ds_sel, ep_ds_sel, join="inner")
        print(f"[INFO] Found {ar_ds_sel.time.size} common time steps and {ar_ds_sel.lat.size}x{ar_ds_sel.lon.size} spatial points.")
        print(f"\n[INFO] Starting spatial diagnostics...")
        run_spatial_diagnostics(ar_ds_sel, ep_ds_sel, output_dir=output_dir)
        print("\nAll diagnostics completed. Check the output directory for results for the full domain.")
    except FileNotFoundError as e:
        print(f"Error: The file {e.filename} was not found. Please check your paths.")
    except Exception as e:
        print(f"An unexpected error occurred during setup: {e}")
