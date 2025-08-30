import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from netCDF4 import num2date, date2num
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
import os
from pandas import to_datetime
import pandas as pd

# 打开数据集
ar_ds = nc.Dataset('G:/ar_analysis/ar_happen.nc')
precip_ds = nc.Dataset('G:/ar_analysis/data/extreme_precipitation_daily.nc')

# 获取时间
ar_time = ar_ds.variables['time'][:]
ar_time_units = ar_ds.variables['time'].units
ar_time_calendar = ar_ds.variables['time'].calendar
ar_dates = num2date(ar_time, units=ar_time_units, calendar=ar_time_calendar)

precip_time = precip_ds.variables['time'][:]
precip_time_units = precip_ds.variables['time'].units
precip_time_calendar = precip_ds.variables['time'].calendar
precip_dates = num2date(precip_time, units=precip_time_units, calendar=precip_time_calendar)

print(f"AR time range: {ar_dates[0]} to {ar_dates[-1]}")
print(f"Precip time range: {precip_dates[0]} to {precip_dates[-1]}")
print(f"Precip time steps: {len(precip_time)}, first 5: {precip_dates[:5]}")

# 时间对齐（统一时间单位）
unified_time_units = "hours since 1970-01-01 00:00:00"
unified_calendar = "proleptic_gregorian"
ar_time_unified = date2num(ar_dates, units=unified_time_units, calendar=unified_calendar)
precip_time_unified = date2num(precip_dates, units=unified_time_units, calendar=unified_calendar)
ar_time_unified = np.round(ar_time_unified, decimals=6)
precip_time_unified = np.round(precip_time_unified, decimals=6)
common_times_unified = np.intersect1d(ar_time_unified, precip_time_unified)

if len(common_times_unified) == 0:
    print("错误：无共同时间点！")
    print("AR time units:", ar_time_units)
    print("AR calendar:", ar_time_calendar)
    print("Precip time units:", precip_time_units)
    print("Precip calendar:", precip_time_calendar)
    print("AR times unified (first 5):", ar_time_unified[:5])
    print("Precip times unified (first 5):", precip_time_unified[:5])
    ar_ds.close()
    precip_ds.close()
    exit()

ar_indices = np.array([np.where(np.round(date2num(ar_dates, unified_time_units, unified_calendar), 6) == t)[0][0] for t in common_times_unified])
precip_indices = np.array([np.where(np.round(date2num(precip_dates, unified_time_units, unified_calendar), 6) == t)[0][0] for t in common_times_unified])
n_time = len(common_times_unified)
common_dates = num2date(common_times_unified, units=unified_time_units, calendar=unified_calendar)
common_dates_pd = to_datetime([d.strftime('%Y-%m-%d %H:%M:%S') for d in common_dates])
print(f"共同时间步长：{n_time}")
print(f"共同时间范围：{common_dates[0]} 到 {common_dates[-1]}")

# 空间范围
lon_min, lon_max = 105, 150
lat_min, lat_max = 15, 50
grid_res = 0.25
lons = np.arange(lon_min, lon_max + grid_res, grid_res)
lats = np.arange(lat_min, lat_max + grid_res, grid_res)
n_lat, n_lon = len(lats), len(lons)

# 裁剪precip数据
precip_lats = precip_ds.variables['latitude'][:]
precip_lons = precip_ds.variables['longitude'][:]
lat_mask = (precip_lats >= lat_min) & (precip_lats <= lat_max)
lon_mask = (precip_lons >= lon_min) & (precip_lons <= lon_max)
lat_indices = np.where(lat_mask)[0]
lon_indices = np.where(lon_mask)[0]
precip_lats = precip_lats[lat_indices]
precip_lons = precip_lons[lon_indices]

# 裁剪AR数据
ar_lats = ar_ds.variables['lat'][:]
ar_lons = ar_ds.variables['lon'][:]
ar_lat_mask = (ar_lats >= lat_min) & (ar_lats <= lat_max)
ar_lon_mask = (ar_lons >= lon_min) & (ar_lons <= lon_max)
ar_lat_indices = np.where(ar_lat_mask)[0]
ar_lon_indices = np.where(ar_lon_mask)[0]
ar_lats = ar_lats[ar_lat_indices]
ar_lons = ar_lons[ar_lon_indices]

# 确保纬度顺序一致（升序：从低到高）
if ar_lats[0] > ar_lats[-1]:
    ar_lats = ar_lats[::-1]
    ar_lat_indices = ar_lat_indices[::-1]
if precip_lats[0] > precip_lats[-1]:
    precip_lats = precip_lats[::-1]
    lat_indices = lat_indices[::-1]

# 验证空间维度一致性
print(f"AR 纬度范围：{ar_lats[0]} 到 {ar_lats[-1]}, 长度：{len(ar_lats)}")
print(f"AR 经度范围：{ar_lons[0]} 到 {ar_lons[-1]}, 长度：{len(ar_lons)}")
print(f"Precip 裁剪后纬度范围：{precip_lats[0]} 到 {precip_lats[-1]}, 长度：{len(precip_lats)}")
print(f"Precip 裁剪后经度范围：{precip_lons[0]} 到 {precip_lons[-1]}, 长度：{len(precip_lons)}")
if not (np.allclose(ar_lats, precip_lats, rtol=1e-5, atol=1e-5) and np.allclose(ar_lons, precip_lons, rtol=1e-5, atol=1e-5)):
    print("错误：AR和Precip的空间网格不一致！")
    print("AR lats (first 5):", ar_lats[:5])
    print("Precip lats (first 5):", precip_lats[:5])
    print("AR lons (first 5):", ar_lons[:5])
    print("Precip lons (first 5):", precip_lons[:5])
    ar_ds.close()
    precip_ds.close()
    exit()

# 预加载数据
print("加载AR和降水数据...")
ar_data = ar_ds.variables['ar_happen'][ar_indices, ar_lat_indices, ar_lon_indices].astype(np.int8)
# 检查是否存在海洋和陆地极端降水标志
if 'extreme_precipitation_flag_ocean' in precip_ds.variables and 'extreme_precipitation_flag_land' in precip_ds.variables:
    # 加载海洋和陆地极端降水标志，合并为单一标志
    precip_flag_ocean = precip_ds.variables['extreme_precipitation_flag_ocean'][precip_indices, lat_indices, lon_indices].astype(np.int8)
    precip_flag_land = precip_ds.variables['extreme_precipitation_flag_land'][precip_indices, lat_indices, lon_indices].astype(np.int8)
    # 合并：如果海洋或陆地任一标志为 1，则视为极端降水事件
    precip_data = np.maximum(precip_flag_ocean, precip_flag_land)
else:
    # 回退到使用单一的极端降水标志
    if 'extreme_precipitation_flag' not in precip_ds.variables:
        raise KeyError("Dataset does not contain 'extreme_precipitation_flag', 'extreme_precipitation_flag_ocean', or 'extreme_precipitation_flag_land'")
    precip_data = precip_ds.variables['extreme_precipitation_flag'][precip_indices, lat_indices, lon_indices].astype(np.int8)

print("数据加载完成")
print(f"AR 数据形状：{ar_data.shape}, 均值：{np.mean(ar_data):.3f}")
print(f"Precip 数据形状：{precip_data.shape}, 均值：{np.mean(precip_data):.3f}")

# 验证数据值
print("AR 数据唯一值：", np.unique(ar_data))
print("Precip 数据唯一值：", np.unique(precip_data))

# 生成季度时间段
start_date = datetime(2003, 12, 31, 18, 0, 0)
end_date = datetime(2024, 12, 31, 18, 0, 0)
quarters = []
current_date = start_date
while current_date < end_date:
    quarter_end = current_date + timedelta(days=90)  # 约3个月
    if quarter_end > end_date:
        quarter_end = end_date
    quarters.append((current_date, quarter_end))
    current_date = quarter_end + timedelta(hours=12)  # 步进12小时以对齐数据

# 计算每季度的概率和Lift
p_extreme_given_ar_list = []
p_extreme_given_no_ar_list = []
lift_list = []
quarter_labels = []
for i, (start, end) in enumerate(tqdm(quarters, desc="Processing quarters")):
    # 选择时间段
    start_time = date2num(start, units=unified_time_units, calendar=unified_calendar)
    end_time = date2num(end, units=unified_time_units, calendar=unified_calendar)
    quarter_idx = (common_times_unified >= start_time) & (common_times_unified <= end_time)
    if not np.any(quarter_idx):
        print(f"警告：{start} 到 {end} 无数据，跳过")
        continue

    quarter_ar_data = ar_data[quarter_idx]
    quarter_precip_data = precip_data[quarter_idx]
    n_time_quarter = quarter_ar_data.shape[0]

    # 列联表
    A = np.sum((quarter_ar_data == 1) & (quarter_precip_data == 1), axis=0)
    B = np.sum((quarter_ar_data == 1) & (quarter_precip_data == 0), axis=0)
    C = np.sum((quarter_ar_data == 0) & (quarter_precip_data == 1), axis=0)
    D = np.sum((quarter_ar_data == 0) & (quarter_precip_data == 0), axis=0)

    # 验证列联表
    total = A + B + C + D
    if not np.allclose(total, n_time_quarter):
        print(f"错误：{start} 到 {end} 列联表总和不等于时间步数！")
        continue

    # 条件概率
    p_extreme_given_ar = np.where((A + B) > 0, A / (A + B), np.nan)
    p_extreme_given_no_ar = np.where((C + D) > 0, C / (C + D), np.nan)
    lift = np.where(p_extreme_given_no_ar > 0, p_extreme_given_ar / p_extreme_given_no_ar, np.nan)

    # 空间平均
    p_extreme_given_ar_list.append(np.nanmean(p_extreme_given_ar))
    p_extreme_given_no_ar_list.append(np.nanmean(p_extreme_given_no_ar))
    lift_list.append(np.nanmean(lift))
    quarter_labels.append(f"{start.strftime('%Y-%m')}-{end.strftime('%m')}")

# 转换为numpy数组
p_extreme_given_ar_list = np.array(p_extreme_given_ar_list)
p_extreme_given_no_ar_list = np.array(p_extreme_given_no_ar_list)
lift_list = np.array(lift_list)

# 最小二乘法拟合
def linear_fit(x, y):
    valid = ~np.isnan(y)
    if np.sum(valid) < 2:
        return np.nan, np.nan
    x_valid = x[valid]
    y_valid = y[valid]
    coeffs = np.polyfit(x_valid, y_valid, 1)  # 线性拟合
    slope, intercept = coeffs
    return slope, intercept

# 时间轴（以年为单位）
time_years = np.array([(start_date + timedelta(days=90 * i)).year + (start_date + timedelta(days=90 * i)).month / 12.0 for i in range(len(quarters))])

# 拟合
slope_p_ar, intercept_p_ar = linear_fit(time_years, p_extreme_given_ar_list)
slope_p_no_ar, intercept_p_no_ar = linear_fit(time_years, p_extreme_given_no_ar_list)
slope_lift, intercept_lift = linear_fit(time_years, lift_list)

# 统计信息
print("统计基础信息...")
print(f"平均 P(Extreme | AR): {np.nanmean(p_extreme_given_ar_list):.3f}")
print(f"平均 P(Extreme | No AR): {np.nanmean(p_extreme_given_no_ar_list):.3f}")
print(f"平均 Lift: {np.nanmean(lift_list):.3f}")
print(f"P(Extreme | AR) 趋势斜率: {slope_p_ar:.4f}, 截距: {intercept_p_ar:.3f}")
print(f"P(Extreme | No AR) 趋势斜率: {slope_p_no_ar:.4f}, 截距: {intercept_p_no_ar:.3f}")
print(f"Lift 趋势斜率: {slope_lift:.4f}, 截距: {intercept_lift:.3f}")

# 可视化折线图
def plot_trend(x, y, slope, intercept, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'o-', label='Data')
    if not np.isnan(slope):
        plt.plot(x, slope * x + intercept, 'r--', label=f'Fit: y = {slope:.4f}x + {intercept:.3f}')
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.title(f'{title}\nfrom {start_date} to {end_date}')
    plt.legend()
    plt.grid(True)

    # 保存图像
    output_dir = "G:/ar_analysis/output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()

# 绘制三张折线图
plot_trend(time_years, p_extreme_given_ar_list, slope_p_ar, intercept_p_ar,
           'P(Extreme Precipitation | AR) Trend', 'P(Extreme | AR)', 'p_extreme_given_ar_trend')
plot_trend(time_years, p_extreme_given_no_ar_list, slope_p_no_ar, intercept_p_no_ar,
           'P(Extreme Precipitation | No AR) Trend', 'P(Extreme | No AR)', 'p_extreme_given_no_ar_trend')
plot_trend(time_years, lift_list, slope_lift, intercept_lift,
           'Lift Trend [P(Extreme | AR) / P(Extreme | No AR)]', 'Lift', 'lift_trend')

# 保存统计信息到文件
output_dir = "G:/ar_analysis/output"
with open(f"{output_dir}/trend_stats.txt", "w") as f:
    f.write(f"Average P(Extreme | AR): {np.nanmean(p_extreme_given_ar_list):.3f}\n")
    f.write(f"Average P(Extreme | No AR): {np.nanmean(p_extreme_given_no_ar_list):.3f}\n")
    f.write(f"Average Lift: {np.nanmean(lift_list):.3f}\n")
    f.write(f"P(Extreme | AR) Trend: slope = {slope_p_ar:.4f}, intercept = {intercept_p_ar:.3f}\n")
    f.write(f"P(Extreme | No AR) Trend: slope = {slope_p_no_ar:.4f}, intercept = {intercept_p_no_ar:.3f}\n")
    f.write(f"Lift Trend: slope = {slope_lift:.4f}, intercept = {intercept_lift:.3f}\n")

# 保存季度数据到CSV
quarter_data = pd.DataFrame({
    'Quarter': quarter_labels,
    'P(Extreme | AR)': p_extreme_given_ar_list,
    'P(Extreme | No AR)': p_extreme_given_no_ar_list,
    'Lift': lift_list
})
quarter_data.to_csv(f"{output_dir}/quarterly_stats.csv", index=False)

# 关闭数据集
ar_ds.close()
precip_ds.close()