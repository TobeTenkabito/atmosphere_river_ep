import xarray as xr
import numpy as np
import pandas as pd
import pygrib
import os
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


def load_and_filter_precipitation(file_path, start_time, end_time):
    """逐时间步加载并过滤降水数据到临时 NetCDF 文件"""
    temp_file = os.path.join(temp_dir, "temp_precip.nc")
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GRIB 文件未找到: {file_path}")

    precip_data = []
    time_values = []
    lats = None
    lons = None

    print("正在过滤 GRIB 数据...")
    grbs = pygrib.open(file_path)
    seen_times = set()
    for grb in grbs:
        if grb.shortName == 'tp' and grb.paramId == 228 and grb.typeOfLevel == 'surface':
            grb_time = pd.to_datetime(f"{grb.dataDate}{grb.dataTime:04d}", format='%Y%m%d%H%M')
            if start_dt <= grb_time <= end_dt and grb_time not in seen_times:
                data = grb.values.astype(np.float32) * 1000  # 米转毫米
                data = np.where(data >= 0, data, 0)
                precip_data.append(data)
                time_values.append(grb_time)
                seen_times.add(grb_time)
                if lats is None:
                    lats, lons = grb.latlons()
    grbs.close()

    if not precip_data:
        raise KeyError(f"在时间范围 {start_time} 到 {end_time} 内未找到 'tp' 数据")

    precip_data = np.array(precip_data)  # 形状: (time, lat, lon)
    time_values = np.array(time_values)
    print(f"过滤得到 {len(time_values)} 个时间步，形状: {precip_data.shape}")

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

    ds.to_netcdf(temp_file, engine='netcdf4')
    print(f"已保存过滤数据到 {temp_file}")
    ds.close()

    with xr.open_dataset(temp_file) as ds_temp:
        precip = ds_temp['tp'].load()
        time_values = ds_temp['time'].values
        print(f"时间步数: {len(time_values)}, 前 5 个: {time_values[:5]}")
        if len(time_values) > 1:
            intervals = np.diff(time_values[:5]).astype('timedelta64[h]')
            print(f"时间间隔 (小时): {intervals}")

    return ds, precip, temp_file


def calculate_extreme_threshold(precip, window_days=15):
    """
    按日历日和格点计算 90th 百分位阈值，使用滑动日历日窗口（仅对非零降水）
    window_days: 滑动窗口大小（奇数，例如15天）
    返回形状为 (day_of_year, lat, lon) 的阈值数组，day_of_year=1到366
    """
    print(f"正在按日历日（{window_days}天窗口）逐格点计算 90th 百分位阈值...")
    ny, nx = precip.shape[1], precip.shape[2]  # lat, lon 维度
    threshold_grid = np.zeros((366, ny, nx), dtype=np.float32)  # 形状: (day_of_year, lat, lon)

    # 提取时间对应的日历日（1到366）
    times = pd.to_datetime(precip['time'].values)
    day_of_year = times.dayofyear  # 形状: (time,)

    # 窗口半宽（前后7天）
    half_window = window_days // 2

    for doy in range(1, 367):  # 1到366
        # 构建滑动窗口的日历日范围
        window_doys = [(doy + i - 1) % 366 + 1 for i in range(-half_window, half_window + 1)]
        # 选择窗口内的所有时间步
        mask = np.isin(day_of_year, window_doys)
        if not mask.any():
            print(f"警告: 日历日 {doy} 窗口无数据，阈值设为0")
            threshold_grid[doy-1] = 0.0
            continue

        # 提取窗口内的降水数据
        window_data = precip[mask].values  # 形状: (window_time, lat, lon)

        for i in range(ny):
            for j in range(nx):
                # 提取该格点窗口内的降水值
                data = window_data[:, i, j]
                non_zero = data[data > 0]  # 仅考虑非零降水
                if non_zero.size > 0:
                    threshold_grid[doy-1, i, j] = np.percentile(non_zero, 90)
                else:
                    threshold_grid[doy-1, i, j] = 0.0  # 无降水数据，阈值为0

    return threshold_grid


def count_precip_events(precip, threshold_grid):
    """
    逐时间步和格点统计极端和非极端降水事件
    threshold_grid: 形状为 (day_of_year, lat, lon) 的阈值数组
    """
    extreme_count = np.zeros(len(precip['time']), dtype=np.int64)  # 每个时间步的总极端事件数
    non_extreme_count = np.zeros(len(precip['time']), dtype=np.int64)  # 每个时间步的非极端事件数
    extreme_flag = np.zeros(precip.shape, dtype=np.int32)  # 形状: (time, lat, lon)
    extreme_count_grid = np.zeros(precip.shape[1:], dtype=np.int64)  # 形状: (lat, lon)

    print("正在统计逐格点按日历日的极端降水事件...")
    day_of_year = pd.to_datetime(precip['time'].values).dayofyear  # 形状: (time,)

    for t in range(len(precip['time'])):
        data = precip.isel(time=t).values  # 形状: (lat, lon)
        doy = day_of_year[t]
        # 使用该时间步对应的日历日阈值
        doy_threshold = threshold_grid[doy-1]  # 形状: (lat, lon)
        extreme_mask = (data >= doy_threshold) & (data > 0)
        non_extreme_mask = (data < doy_threshold) & (data > 0)
        extreme_count[t] = np.sum(extreme_mask)
        non_extreme_count[t] = np.sum(non_extreme_mask)
        extreme_flag[t] = extreme_mask.astype(np.int32)
        extreme_count_grid += extreme_mask.astype(np.int64)

    return extreme_count, non_extreme_count, extreme_flag, extreme_count_grid


def seasonal_analysis(precip, threshold_grid):
    """逐时间步执行季节性极端降水分析，基于按日历日的逐格点阈值"""
    seasons = {
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11],
        'Winter': [12, 1, 2]
    }
    seasonal_extreme = {s: 0 for s in seasons}
    seasonal_sum_precip = {s: 0.0 for s in seasons}
    seasonal_count = {s: 0 for s in seasons}

    time_dim = 'time'
    times = pd.to_datetime(precip[time_dim].values)
    day_of_year = times.dayofyear
    months = times.month

    for t in range(len(precip[time_dim])):
        data = precip.isel(time=t).values  # 形状: (lat, lon)
        doy = day_of_year[t]
        month = months[t]
        doy_threshold = threshold_grid[doy-1]  # 使用对应日历日的阈值
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

    for season in seasons:
        print(f"{season} 极端事件: {seasonal_extreme[season]}, 数据点: {seasonal_count[season]}")

    return seasonal_extreme, seasonal_avg_precip


def save_extreme_precipitation(precip, extreme_flag, extreme_count_grid, threshold_grid, output_path, start_time, end_time):
    """
    将逐格点按日历日的极端降水数据、阈值和极端降水量保存到 NetCDF 文件（符合 CF 标准）
    threshold_grid: 形状为 (day_of_year, lat, lon) 的阈值数组
    """
    # 创建极端降水量数组：仅保留 extreme_flag == 1 的降水值，其余为 NaN
    extreme_precip_amount = np.where(extreme_flag == 1, precip.values, np.nan)

    ds_out = xr.Dataset(
        {
            'extreme_precipitation_flag': (
                ['time', 'latitude', 'longitude'],
                extreme_flag,
                {
                    'long_name': 'Extreme precipitation occurrence flag (daily grid-wise)',
                    'standard_name': 'precipitation_amount_above_daily_grid_threshold',
                    'description': '1 indicates precipitation >= daily grid-wise 90th percentile (15-day window); 0 otherwise',
                    'units': '1',
                    'comment': 'Boolean mask: 1 = extreme precipitation event based on daily grid-wise threshold'
                }
            ),
            'extreme_precipitation_amount': (
                ['time', 'latitude', 'longitude'],
                extreme_precip_amount,
                {
                    'long_name': 'Extreme precipitation amount (daily grid-wise)',
                    'standard_name': 'extreme_precipitation_amount',
                    'description': 'Precipitation amount for extreme events (>= daily grid-wise 90th percentile, 15-day window); NaN for non-extreme events',
                    'units': 'mm',
                    'comment': 'Precipitation values where extreme_flag is 1, otherwise NaN'
                }
            ),
            'extreme_precipitation_count': (
                ['latitude', 'longitude'],
                extreme_count_grid,
                {
                    'long_name': 'Total count of extreme precipitation events (daily grid-wise)',
                    'standard_name': 'number_of_extreme_precipitation_events',
                    'units': '1',
                    'description': 'Total number of time steps where precipitation exceeded daily grid-wise 90th percentile'
                }
            ),
            'threshold_90th': (
                ['day_of_year', 'latitude', 'longitude'],
                threshold_grid,
                {
                    'long_name': 'Daily 90th percentile precipitation threshold per grid',
                    'standard_name': 'precipitation_amount_threshold',
                    'units': 'mm',
                    'description': 'Grid-wise 90th percentile of non-zero precipitation values for each day of year (15-day window)'
                }
            )
        },
        coords={
            'time': precip['time'],
            'latitude': precip['latitude'],
            'longitude': precip['longitude'],
            'day_of_year': ('day_of_year', np.arange(1, 367), {'long_name': 'Day of year', 'units': '1'})
        },
        attrs={
            'title': 'Daily grid-wise extreme precipitation dataset',
            'summary': 'Flag, count, amount, and daily threshold of extreme precipitation events (>= daily grid-wise 90th percentile, 15-day window)',
            'time_coverage_start': str(start_time),
            'time_coverage_end': str(end_time),
            'institution': 'Your Institution Name',
            'source': 'ERA5 total precipitation data processed using xarray and pygrib',
            'history': f'Created {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'references': '',
            'Conventions': 'CF-1.6'
        }
    )

    ds_out.to_netcdf(output_path)
    print(f"✅ 已保存逐格点按日历日的极端降水数据（包含降水量）到：{output_path}")


def main():
    file_path = r"G:\precipitation\precipitation.grib"
    output_path = r"G:\ar_analysis\data\extreme_precipitation.nc"  # 修改输出文件名

    try:
        intervals, interval_info = check_time_intervals(file_path)
        print(interval_info)
        if intervals is None:
            raise ValueError("时间步不足，无法继续")

        start_time = "1983-12-31 18:00:00"
        end_time = "2024-12-31 6:00:00"

        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        if start_dt >= end_dt:
            raise ValueError("结束时间必须晚于开始时间")

        # 加载并过滤数据
        ds, precip, temp_file = load_and_filter_precipitation(file_path, start_time, end_time)

        time_dim = 'time'
        min_time = pd.to_datetime(ds[time_dim].values[0])
        max_time = pd.to_datetime(ds[time_dim].values[-1])
        if start_dt < min_time or end_dt > max_time:
            raise ValueError(f"时间范围必须在 {min_time} 到 {max_time} 之间")

        # 计算逐格点按日历日的阈值（15天窗口）
        threshold_grid = calculate_extreme_threshold(precip, window_days=15)
        print(f"逐格点按日历日阈值形状: {threshold_grid.shape}, 示例值 (DOY 1, 0, 0): {threshold_grid[0, 0, 0]:.2f} mm")

        # 统计逐格点按日历日的事件
        extreme_count, non_extreme_count, extreme_flag, extreme_count_grid = count_precip_events(precip, threshold_grid)

        # 季节性分析
        time_diff = end_dt - start_dt
        seasonal_data = None
        if time_diff <= pd.Timedelta(days=5 * 365):
            seasonal_extreme, seasonal_avg_precip = seasonal_analysis(precip, threshold_grid)
            seasonal_data = (seasonal_extreme, seasonal_avg_precip)

        # 保存结果
        save_extreme_precipitation(precip, extreme_flag, extreme_count_grid, threshold_grid, output_path, start_time, end_time)

        # 打印摘要
        print(f"总极端降水事件数: {extreme_count.sum():.0f}")
        print(f"总非极端降水事件数: {non_extreme_count.sum():.0f}")
        if seasonal_data:
            seasonal_extreme, seasonal_avg_precip = seasonal_data
            print("\n季节性极端事件:")
            for season, count in seasonal_extreme.items():
                print(f"{season}: {count:.0f} 事件")
            print("\n季节性平均降水量 (毫米):")
            for season, precip in seasonal_avg_precip.items():
                print(f"{season}: {precip:.2f} 毫米")

        # 清理临时文件
        os.remove(temp_file)
        print(f"已删除临时文件: {temp_file}")

    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()