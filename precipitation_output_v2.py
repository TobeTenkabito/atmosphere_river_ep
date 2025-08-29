# 本代码采用滑动窗口法计算极端降水
import xarray as xr
import numpy as np
import pandas as pd
import pygrib
import os
import gc
import warnings

# 忽略警告以保持输出简洁
warnings.filterwarnings('ignore')

# 创建目录
output_dir = r"G:\ar_analysis\data"
temp_dir = r"G:\ar_analysis\temp"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)


def check_time_intervals(file_path):
    """检查 GRIB 文件中的时间间隔并返回唯一间隔"""
    try:
        grbs = pygrib.open(file_path)
        times = []
        for grb in grbs:
            if grb.shortName == 'tp' and grb.paramId == 228 and grb.typeOfLevel == 'surface':
                grb_time = pd.to_datetime(f"{grb.dataDate}{grb.dataTime:04d}", format='%Y%m%d%H%M')
                times.append(grb_time)
        grbs.close()
        times = sorted(set(times))
        if len(times) < 2:
            return None, "时间步不足以确定间隔"
        intervals = np.diff(times).astype('timedelta64[h]')
        unique_intervals = np.unique(intervals)
        min_time = min(times)
        max_time = max(times)
        return unique_intervals, f"时间范围: {min_time} 到 {max_time}, 间隔 (小时): {unique_intervals}"
    except Exception as e:
        return None, f"检查间隔时出错: {str(e)}"


def _process_chunk(file_path, chunk_start, chunk_end, temp_dir):
    """
    处理单个时间块（例如一个月）的数据，包括过滤、插值和保存。
    返回插值后的 xarray Dataset。
    """
    chunk_start_str = chunk_start.strftime('%Y%m%d%H%M')
    chunk_end_str = chunk_end.strftime('%Y%m%d%H%M')
    print(f"-> 正在处理时间段: {chunk_start_str} 至 {chunk_end_str}")

    precip_data = []
    time_values = []
    lats = None
    lons = None

    data_dict = {}
    grbs = pygrib.open(file_path)
    for grb in grbs:
        if grb.shortName == 'tp' and grb.paramId == 228 and grb.typeOfLevel == 'surface':
            grb_time = pd.to_datetime(f"{grb.dataDate}{grb.dataTime:04d}", format='%Y%m%d%H%M')
            if chunk_start <= grb_time <= chunk_end:
                data = grb.values.astype(np.float32) * 1000
                data = np.where(data >= 0, data, 0)
                data_dict[grb_time] = data
                if lats is None:
                    lats, lons = grb.latlons()
    grbs.close()

    if not data_dict:
        print(f"   警告: 在此时间段内未找到 'tp' 数据，跳过...")
        return None

    sorted_times = sorted(data_dict.keys())
    precip_data = np.array([data_dict[t] for t in sorted_times])
    time_values = np.array(sorted_times)

    print(f"   过滤得到 {len(time_values)} 个时间步，形状: {precip_data.shape}")

    lat_1d = lats[:, 0]
    lon_1d = lons[0, :]

    ds = xr.Dataset(
        {
            'tp': (['time', 'latitude', 'longitude'], precip_data),
        },
        coords={
            'time': time_values,
            'latitude': ('latitude', lat_1d),
            'longitude': ('longitude', lon_1d)
        }
    )

    # 线性插值到 6 小时
    new_time = pd.date_range(start=ds.time.values[0],
                             end=ds.time.values[-1],
                             freq='6H')

    ds_interp = ds.interp(time=new_time, method='linear', kwargs={'fill_value': 'extrapolate'})
    print(f"   插值后得到 {len(ds_interp.time.values)} 个时间步，新形状: {ds_interp['tp'].shape}")

    return ds_interp


def load_and_filter_precipitation(file_path, start_time, end_time, temp_dir):
    """
    分块处理降水数据，逐月保存到临时文件，最后通过 open_mfdataset 自动拼接。
    """
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)

    # 确保临时目录存在
    os.makedirs(temp_dir, exist_ok=True)

    # 时间分块：逐月
    time_chunks = pd.date_range(start_dt.to_period('M').start_time, end_dt, freq='M')
    time_chunks = time_chunks.union([end_dt])

    saved_files = []

    for i in range(len(time_chunks) - 1):
        chunk_start = time_chunks[i]
        chunk_end = time_chunks[i + 1]

        extended_start = chunk_start - pd.Timedelta(days=1)
        extended_end = chunk_end + pd.Timedelta(days=1)

        processed_chunk = _process_chunk(file_path, extended_start, extended_end, temp_dir)

        if processed_chunk is not None:
            # ⚠️ 裁剪严格到当前月，避免时间重叠
            processed_chunk = processed_chunk.sel(time=slice(chunk_start, chunk_end))

            # 保存到单独文件
            out_file = os.path.join(temp_dir, f"precip_{chunk_start:%Y%m}.nc")
            processed_chunk.to_netcdf(out_file, mode="w", engine="netcdf4")
            saved_files.append(out_file)

            print(f"✅ 已保存: {out_file}")

    # 用 open_mfdataset 拼接，保证时间唯一且懒加载
    ds = xr.open_mfdataset(
        saved_files,
        combine="nested",  # 按顺序拼接
        concat_dim="time",  # 在时间维度上拼接
        parallel=True,
        chunks={"time": 100}
    )

    print("🎉 数据已成功拼接（懒加载，不会占满内存）")
    return ds


def calculate_extreme_threshold(precip, window_days=15):
    print(f"正在按日历日（{window_days}天窗口）逐格点计算 90th 百分位阈值...")
    ny, nx = precip.shape[1], precip.shape[2]
    threshold_grid = np.zeros((366, ny, nx), dtype=np.float32)
    times = pd.to_datetime(precip['time'].values)
    day_of_year = times.dayofyear
    half_window = window_days // 2

    for doy in range(1, 367):
        window_doys = [(doy + i - 1) % 366 + 1 for i in range(-half_window, half_window + 1)]
        mask = np.isin(day_of_year, window_doys)
        if not mask.any():
            threshold_grid[doy-1] = 0.0
            continue
        window_data = precip[mask].values
        for i in range(ny):
            for j in range(nx):
                data = window_data[:, i, j]
                non_zero = data[data > 0]
                if non_zero.size > 0:
                    threshold_grid[doy-1, i, j] = np.percentile(non_zero, 95)
                else:
                    threshold_grid[doy-1, i, j] = 0.0

    return threshold_grid


def count_precip_events(precip, threshold_grid):
    extreme_count = np.zeros(len(precip['time']), dtype=np.int64)
    non_extreme_count = np.zeros(len(precip['time']), dtype=np.int64)
    extreme_flag = np.zeros(precip.shape, dtype=np.int32)
    extreme_count_grid = np.zeros(precip.shape[1:], dtype=np.int64)

    day_of_year = pd.to_datetime(precip['time'].values).dayofyear
    for t in range(len(precip['time'])):
        data = precip.isel(time=t).values
        doy = day_of_year[t]
        doy_threshold = threshold_grid[doy-1]
        extreme_mask = (data >= doy_threshold) & (data > 0)
        non_extreme_mask = (data < doy_threshold) & (data > 0)
        extreme_count[t] = np.sum(extreme_mask)
        non_extreme_count[t] = np.sum(non_extreme_mask)
        extreme_flag[t] = extreme_mask.astype(np.int32)
        extreme_count_grid += extreme_mask.astype(np.int64)
    return extreme_count, non_extreme_count, extreme_flag, extreme_count_grid


def seasonal_analysis(precip, threshold_grid):
    seasons = {
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11],
        'Winter': [12, 1, 2]
    }
    seasonal_extreme = {s: 0 for s in seasons}
    seasonal_sum_precip = {s: 0.0 for s in seasons}
    seasonal_count = {s: 0 for s in seasons}

    times = pd.to_datetime(precip['time'].values)
    day_of_year = times.dayofyear
    months = times.month

    for t in range(len(precip['time'])):
        data = precip.isel(time=t).values
        doy = day_of_year[t]
        month = months[t]
        doy_threshold = threshold_grid[doy-1]
        for season, season_months in seasons.items():
            if month in season_months:
                extreme_mask = (data >= doy_threshold) & (data > 0)
                seasonal_extreme[season] += np.sum(extreme_mask)
                seasonal_sum_precip[season] += np.sum(data)
                seasonal_count[season] += np.sum(data > 0)
                break

    seasonal_avg_precip = {
        season: seasonal_sum_precip[season] / seasonal_count[season] if seasonal_count[season] > 0 else 0.0
        for season in seasons
    }
    return seasonal_extreme, seasonal_avg_precip


def save_extreme_precipitation(precip, extreme_flag, extreme_count_grid, threshold_grid, output_path, start_time, end_time):
    extreme_precip_amount = np.where(extreme_flag == 1, precip.values, np.nan)
    ds_out = xr.Dataset(
        {
            'extreme_precipitation_flag': (['time', 'latitude', 'longitude'], extreme_flag),
            'extreme_precipitation_amount': (['time', 'latitude', 'longitude'], extreme_precip_amount),
            'extreme_precipitation_count': (['latitude', 'longitude'], extreme_count_grid),
            'threshold_90th': (['day_of_year', 'latitude', 'longitude'], threshold_grid)
        },
        coords={
            'time': precip['time'],
            'latitude': precip['latitude'],
            'longitude': precip['longitude'],
            'day_of_year': ('day_of_year', np.arange(1, 367))
        },
        attrs={
            'title': 'Daily grid-wise extreme precipitation dataset',
            'summary': 'Flag, count, amount, and daily threshold of extreme precipitation events',
            'time_coverage_start': str(start_time),
            'time_coverage_end': str(end_time),
            'institution': 'Your Institution Name',
            'source': 'ERA5 total precipitation data',
            'history': f'Created {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'Conventions': 'CF-1.6'
        }
    )
    ds_out.to_netcdf(output_path)
    print(f"✅ 已保存结果到: {output_path}")


def main():
    file_path = r"G:\precipitation\precipitation.grib"
    output_path = r"G:\ar_analysis\data\extreme_precipitation.nc"
    temp_file = os.path.join(temp_dir, "temp_precip_combined")

    try:
        intervals, interval_info = check_time_intervals(file_path)
        print(interval_info)
        if intervals is None:
            raise ValueError("时间步不足，无法继续")

        start_time = "1983-12-31 18:00:00"
        end_time = "2024-12-31 06:00:00"

        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        if start_dt >= end_dt:
            raise ValueError("结束时间必须晚于开始时间")

        # 加载并过滤数据（逐月写入）
        ds = load_and_filter_precipitation(file_path, start_time, end_time, temp_file)
        precip = ds['tp']

        # 计算逐格点按日历日阈值
        threshold_grid = calculate_extreme_threshold(precip, window_days=15)

        # 统计逐格点按日历日的事件
        extreme_count, non_extreme_count, extreme_flag, extreme_count_grid = count_precip_events(precip, threshold_grid)

        # 季节性分析（仅在时间跨度小于5年时运行）
        time_diff = end_dt - start_dt
        if time_diff <= pd.Timedelta(days=5 * 365):
            seasonal_extreme, seasonal_avg_precip = seasonal_analysis(precip, threshold_grid)
            print("季节性极端事件:", seasonal_extreme)
            print("季节性平均降水量:", seasonal_avg_precip)

        # 保存结果
        save_extreme_precipitation(precip, extreme_flag, extreme_count_grid, threshold_grid,
                                   output_path, start_time, end_time)

        print(f"总极端降水事件数: {extreme_count.sum():.0f}")
        print(f"总非极端降水事件数: {non_extreme_count.sum():.0f}")

        # 清理临时文件
        ds.close()
        os.remove(temp_file)
        print(f"已删除临时文件: {temp_file}")

    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()
