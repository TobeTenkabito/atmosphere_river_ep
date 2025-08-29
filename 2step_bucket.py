import matplotlib
matplotlib.use('TkAgg')
import netCDF4 as nc
import numpy as np
from netCDF4 import num2date, date2num
from datetime import datetime
from scipy.stats import fisher_exact, chi2
import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =============================
# 可调参数
# =============================
grid_aggregate_size = 1
sig_level = 0.05
lift_cap_percentile = 95
lift_max_val = 10
output_dir = "G:/ar_analysis/output_1"
os.makedirs(output_dir, exist_ok=True)

# 是否启用控制桶 (新增 "AO")
control_modes = ["ENSO", "MJO", "AO"]
do_heterogeneity_test = True

# 定义区域经纬度
REGIONS = {
    "South_China": {'lon_min': 105, 'lon_max': 125, 'lat_min': 18, 'lat_max': 32},
    "North_China": {'lon_min': 105, 'lon_max': 125, 'lat_min': 32, 'lat_max': 42},
    "Northeast_Asia": {'lon_min': 120, 'lon_max': 140, 'lat_min': 40, 'lat_max': 58},
    "Japan_Korea": {'lon_min': 125, 'lon_max': 145, 'lat_min': 30, 'lat_max': 45}
}


# =============================
# 载入 ENSO, MJO & AO 数据
# =============================
def load_enso():
    enso_file = "G:/precipitation/nino3.4.csv"
    # 从第二行开始读取数据，并将第一列设为索引
    df = pd.read_csv(enso_file, header=1, index_col=0)
    # 将索引转换为 datetime 对象，处理多种日期格式
    df.index = pd.to_datetime(df.index, errors='coerce', format='mixed')
    df = df.dropna(subset=[df.columns[0]])
    # 获取 nino3.4 值
    values = df.iloc[:, 0].astype(float)
    # 根据值进行桶分类
    bucket = pd.Series(index=values.index, dtype="object")
    bucket[values > 0.5] = "ElNino"
    bucket[values < -0.5] = "LaNina"
    bucket[(values >= -0.5) & (values <= 0.5)] = "Neutral"
    return bucket


def load_mjo():
    mjo_file = "G:/precipitation/MJO.CSV"
    # 从第三行开始读取数据
    df = pd.read_csv(mjo_file, header=2)
    # 确保列名正确
    df.columns = ['year', 'month', 'day', 'RMM1', 'RMM2', 'RMM_phase', 'RMM_amplitude', 'RMM_weight', "Column9",
                  "Column10"]
    # 合并年、月、日列创建时间戳索引
    df['time'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    df = df.dropna(subset=['time'])
    df = df.set_index('time')
    # 桶分类: inactive / active-1to4 / active-5to8
    amp = df['RMM_amplitude']
    phase = df['RMM_phase']
    bucket = pd.Series(index=df.index, dtype="object")
    bucket[amp < 1] = "Inactive"
    bucket[(amp >= 1) & (phase.isin([1, 2, 3, 4]))] = "Active_1to4"
    bucket[(amp >= 1) & (phase.isin([5, 6, 7, 8]))] = "Active_5to8"
    return bucket


# =============================
# 新增: 载入 AO 数据函数
# =============================
def load_ao():
    """
    加载 AO.csv 文件并根据 index 值进行分桶
    """
    ao_file = "G:/precipitation/AO.CSV"  # 请确保 AO.csv 文件在此脚本的同一目录下，或提供完整路径
    df = pd.read_csv(ao_file)
    # 从 'year' 和 'month' 列创建 datetime 索引
    df['time'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')
    df = df.set_index('time')

    values = df['index'].astype(float)
    bucket = pd.Series(index=values.index, dtype="object")
    bucket[values > 0.5] = "Positive"
    bucket[values < -0.5] = "Negative"
    bucket[(values >= -0.5) & (values <= 0.5)] = "Neutral"
    return bucket


def get_bucket_labels(dates_series):
    """
    根据日期时间序列获取对应的 ENSO, MJO 和 AO 桶标签。
    """
    labels = {}
    if "ENSO" in control_modes:
        enso_bucket = load_enso()
        # 使用最近的时间索引进行对齐
        labels["ENSO"] = enso_bucket.reindex(dates_series, method='nearest').values
    if "MJO" in control_modes:
        mjo_bucket = load_mjo()
        labels["MJO"] = mjo_bucket.reindex(dates_series, method='nearest').values
    # 新增: 获取 AO 桶标签
    if "AO" in control_modes:
        ao_bucket = load_ao()
        labels["AO"] = ao_bucket.reindex(dates_series, method='nearest').values
    return labels


# =============================
# 核心统计函数
# =============================
def compute_stats(ar_mask, pr_mask):
    """
    计算统计指标 (OR, lift, p-value等) 并新增 log(OR) 的标准误。
    """
    A = np.sum((ar_mask == 1) & (pr_mask == 1))
    B = np.sum((ar_mask == 1) & (pr_mask == 0))
    C = np.sum((ar_mask == 0) & (pr_mask == 1))
    D = np.sum((ar_mask == 0) & (pr_mask == 0))
    table = [[A, B], [C, D]]

    # 防止除以零，确保所有值都存在
    if (A + B) == 0 or (C + D) == 0 or A == 0 or B == 0 or C == 0 or D == 0:
        return {"OR": np.nan, "lift": np.nan, "pval": 1, "table": (A, B, C, D), "log_OR": np.nan, "SE": np.nan}

    OR = (A * D) / (B * C)
    p_ext_ar = A / (A + B)
    p_ext_no_ar = C / (C + D)
    lift = p_ext_ar / p_ext_no_ar if p_ext_no_ar > 0 else np.nan
    _, pval = fisher_exact(table)

    # 计算 log(OR) 及其标准误
    log_OR = np.log(OR)
    SE = np.sqrt(1 / A + 1 / B + 1 / C + 1 / D)

    return {"OR": OR, "lift": lift, "pval": pval, "table": (A, B, C, D), "log_OR": log_OR, "SE": SE}


def run_analysis(ar_data_subset, precip_data_subset, ar_dates_subset, bucket_labels):
    """
    对给定的数据集子集运行分析 (已更新以处理 AO 桶)
    """
    results = {}

    if not bucket_labels:
        results['overall'] = compute_stats(ar_data_subset, precip_data_subset)
        return results

    # 动态生成所有桶的组合
    import itertools

    active_buckets = {k: np.unique(v) for k, v in bucket_labels.items()}
    bucket_names = list(active_buckets.keys())
    bucket_values_combinations = list(itertools.product(*active_buckets.values()))

    for combo in bucket_values_combinations:
        mask = np.full(len(ar_data_subset), True)
        label_parts = []
        for i, bucket_name in enumerate(bucket_names):
            mask &= (bucket_labels[bucket_name] == combo[i])
            label_parts.append(f"{bucket_name}={combo[i]}")

        label = ", ".join(label_parts)
        if np.any(mask):
            results[label] = compute_stats(ar_data_subset[mask], precip_data_subset[mask])

    if do_heterogeneity_test and len(results) > 1:
        ORs = [r["OR"] for r in results.values() if "OR" in r and not np.isnan(r["OR"])]
        if len(ORs) > 1:
            meanOR = np.mean(ORs)
            Q = np.sum((ORs - meanOR) ** 2)
            df = len(ORs) - 1
            pQ = 1 - chi2.cdf(Q, df)
            I2 = max(0, (Q - df) / Q) * 100 if Q > df else 0
            results["heterogeneity"] = {"Q": Q, "pQ": pQ, "I2": I2}

    return results


# =============================
# 逻辑回归与交互项分析
# =============================
def run_logistic_regression(ar_data, precip_data, dates, enso_bucket, mjo_bucket, ao_bucket):
    """
    执行逻辑回归以进行正式的交互项检验 (已更新以包含 AO)
    """
    print("\nRunning Logistic Regression...")

    # 将 cftime 日期对象转换为标准的 datetime 对象
    dates_dt = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dates]

    # 将数据展平为一维
    ar_flat = ar_data.flatten()
    precip_flat = precip_data.flatten()

    # 内存优化：随机子采样数据
    sample_size = int(len(ar_flat) * 0.05)
    if sample_size > 1000000:
        sample_size = 1000000

    sample_indices = np.random.choice(len(ar_flat), sample_size, replace=False) if len(
        ar_flat) > sample_size else np.arange(len(ar_flat))

    # 使用采样后的数据创建 DataFrame
    df_data = pd.DataFrame({'EP': precip_flat[sample_indices], 'AR': ar_flat[sample_indices]})

    # 为每个数据点添加日期和时间信息
    dates_expanded = np.repeat(dates_dt, ar_data.shape[1] * ar_data.shape[2])
    df_data['date'] = dates_expanded[sample_indices]

    # 添加 ENSO, MJO 和 AO 桶
    if "ENSO" in control_modes:
        df_data['ENSO'] = enso_bucket.reindex(df_data['date'], method='nearest').values
    if "MJO" in control_modes:
        df_data['MJO'] = mjo_bucket.reindex(df_data['date'], method='nearest').values
    if "AO" in control_modes:
        df_data['AO'] = ao_bucket.reindex(df_data['date'], method='nearest').values

    # 创建月度和趋势变量
    df_data['month'] = df_data['date'].dt.month
    df_data['trend'] = df_data.index

    # 将分类变量转换为独热编码
    dummy_cols = [mode for mode in control_modes if mode in df_data.columns]
    df_data = pd.get_dummies(df_data, columns=dummy_cols, drop_first=True, dtype=int)

    # 动态构建交互项和自变量列表
    exog_vars = ['AR', 'trend']
    interaction_pvalues = {}

    # ENSO 交互项
    for col in [c for c in df_data.columns if c.startswith('ENSO_')]:
        interaction_term = f'AR_{col}'
        df_data[interaction_term] = df_data['AR'] * df_data[col]
        exog_vars.extend([col, interaction_term])

    # MJO 交互项
    for col in [c for c in df_data.columns if c.startswith('MJO_')]:
        interaction_term = f'AR_{col}'
        df_data[interaction_term] = df_data['AR'] * df_data[col]
        exog_vars.extend([col, interaction_term])

    # 新增: AO 交互项
    for col in [c for c in df_data.columns if c.startswith('AO_')]:
        interaction_term = f'AR_{col}'
        df_data[interaction_term] = df_data['AR'] * df_data[col]
        exog_vars.extend([col, interaction_term])

    # 添加月度独热编码
    df_data = pd.get_dummies(df_data, columns=['month'], drop_first=True, dtype=int)
    month_dummies = [col for col in df_data.columns if col.startswith('month_')]
    exog_vars.extend(month_dummies)

    # 添加常数项并移除重复项
    exog_vars = sorted(list(set(exog_vars)))
    df_data = sm.add_constant(df_data, has_constant='add')
    exog_vars.insert(0, 'const')

    # 拟合逻辑回归模型
    model = sm.Logit(df_data['EP'], df_data[exog_vars])
    result = model.fit(disp=False)

    # 打印回归结果表格
    print(result.summary())

    # 提取交互项的p值
    print("\n交互项的Wald检验 P值:")
    for var in result.pvalues.index:
        if var.startswith('AR_'):
            interaction_pvalues[var] = result.pvalues[var]
            print(f"  {var}: {interaction_pvalues[var]:.4f}")

    # 绘制边际效应
    num_plots = len(control_modes)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6))
    if num_plots == 1: axes = [axes]  # 确保 axes 是可迭代的

    plot_idx = 0

    # ENSO 边际效应
    if "ENSO" in control_modes:
        enso_or = {'ElNino': np.exp(result.params['AR'])}  # Base case
        if 'AR_ENSO_LaNina' in result.params: enso_or['LaNina'] = np.exp(
            result.params['AR'] + result.params['AR_ENSO_LaNina'])
        if 'AR_ENSO_Neutral' in result.params: enso_or['Neutral'] = np.exp(
            result.params['AR'] + result.params['AR_ENSO_Neutral'])

        sorted_keys = sorted(enso_or.keys())
        axes[plot_idx].bar(sorted_keys, [enso_or[k] for k in sorted_keys])
        axes[plot_idx].set_title('AR-Precipitation OR by ENSO Phase')
        axes[plot_idx].set_ylabel('Odds Ratio')
        axes[plot_idx].axhline(1, color='red', linestyle='--')
        plot_idx += 1

    # MJO 边际效应
    if "MJO" in control_modes:
        mjo_or = {'Inactive': np.exp(result.params['AR'])}  # Base case
        if 'AR_MJO_Active_1to4' in result.params: mjo_or['Active_1to4'] = np.exp(
            result.params['AR'] + result.params['AR_MJO_Active_1to4'])
        if 'AR_MJO_Active_5to8' in result.params: mjo_or['Active_5to8'] = np.exp(
            result.params['AR'] + result.params['AR_MJO_Active_5to8'])

        sorted_keys = sorted(mjo_or.keys())
        axes[plot_idx].bar(sorted_keys, [mjo_or[k] for k in sorted_keys])
        axes[plot_idx].set_title('AR-Precipitation OR by MJO Phase')
        axes[plot_idx].set_ylabel('Odds Ratio')
        axes[plot_idx].axhline(1, color='red', linestyle='--')
        plot_idx += 1

    # 新增: AO 边际效应
    if "AO" in control_modes:
        # 假设 'Negative' 是 get_dummies(drop_first=True) 后的基准组
        ao_or = {'Negative': np.exp(result.params['AR'])}
        if 'AR_AO_Positive' in result.params: ao_or['Positive'] = np.exp(
            result.params['AR'] + result.params['AR_AO_Positive'])
        if 'AR_AO_Neutral' in result.params: ao_or['Neutral'] = np.exp(
            result.params['AR'] + result.params['AR_AO_Neutral'])

        sorted_keys = sorted(ao_or.keys())
        axes[plot_idx].bar(sorted_keys, [ao_or[k] for k in sorted_keys])
        axes[plot_idx].set_title('AR-Precipitation OR by AO Phase')
        axes[plot_idx].set_ylabel('Odds Ratio')
        axes[plot_idx].axhline(1, color='red', linestyle='--')
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "marginal_effects.png"))
    print(f"\n边际效应图已保存至: {os.path.join(output_dir, 'marginal_effects.png')}")
    plt.show()


# =============================
# 区域异质性分析
# =============================
def run_regional_analysis(ar_data, precip_data, ar_dates_sel, ar_lats, ar_lons):
    """
    按区域进行分析，并为每个区域生成森林图
    """
    print("\nRunning regional analysis...")

    # 将 cftime 日期对象转换为标准的 datetime 对象
    ar_dates_sel_dt = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in ar_dates_sel]
    bucket_labels = get_bucket_labels(pd.to_datetime(ar_dates_sel_dt))

    for region_name, region_coords in REGIONS.items():
        print(f"\n--- Analyzing Region: {region_name} ---")

        # 使用布尔掩码直接进行索引
        lat_mask = (ar_lats >= region_coords['lat_min']) & (ar_lats <= region_coords['lat_max'])
        lon_mask = (ar_lons >= region_coords['lon_min']) & (ar_lons <= region_coords['lon_max'])

        if not np.any(lat_mask) or not np.any(lon_mask):
            print(f"  Warning: No grid points found for region {region_name}, skipping.")
            continue

        region_ar_data = ar_data[:, lat_mask, :][:, :, lon_mask]
        region_precip_data = precip_data[:, lat_mask, :][:, :, lon_mask]

        if region_ar_data.size == 0 or region_precip_data.size == 0:
            print(f"  Warning: No data found for region {region_name}, skipping.")
            continue

        # 运行分析并绘制森林图
        region_results = run_analysis(region_ar_data, region_precip_data, ar_dates_sel, bucket_labels)
        plot_forest_plot(region_results, title=f"Forest Plot for {region_name}",
                         filename=f"forest_plot_{region_name}.png")


def plot_forest_plot(results, title="Forest Plot of Odds Ratios", filename="forest_plot.png"):
    """
    生成并保存森林图
    """
    subgroup_results = {k: v for k, v in results.items() if k not in ["overall", "heterogeneity"]}

    if not subgroup_results:
        print("No subgroups to plot.")
        return

    plot_data = {
        k: v for k, v in subgroup_results.items()
        if "OR" in v and not np.isnan(v["OR"]) and "SE" in v and not np.isnan(v["SE"])
    }

    if not plot_data:
        print("No valid data to plot.")
        return

    labels = list(plot_data.keys())
    log_ORs = np.array([v["log_OR"] for v in plot_data.values()])
    SEs = np.array([v["SE"] for v in plot_data.values()])

    lower_bounds = np.exp(log_ORs - 1.96 * SEs)
    upper_bounds = np.exp(log_ORs + 1.96 * SEs)
    ORs = np.exp(log_ORs)

    sort_idx = np.argsort(ORs)[::-1]
    ORs = ORs[sort_idx]
    labels = [labels[i] for i in sort_idx]
    lower_bounds = lower_bounds[sort_idx]
    upper_bounds = upper_bounds[sort_idx]

    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.4)))

    y_pos = np.arange(len(labels))
    ax.scatter(ORs, y_pos, color='blue', zorder=2)
    ax.hlines(y_pos, lower_bounds, upper_bounds, color='gray', linestyle='-', zorder=1)

    ax.axvline(1, color='black', linestyle='--', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Odds Ratio (OR)")
    ax.set_title(title)
    ax.set_xscale('log')
    ax.grid(True, which="both", axis="x", linestyle='--', linewidth=0.5)

    if "heterogeneity" in results:
        het_stats = results["heterogeneity"]
        text_str = (
            f"Heterogeneity Test:\n"
            f"Cochran's Q = {het_stats['Q']:.2f}\n"
            f"p-value = {het_stats['pQ']:.3f}\n"
            f"I² = {het_stats['I2']:.1f}%"
        )
        ax.text(1.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Forest plot saved to {os.path.join(output_dir, filename)}")
    plt.show()


# =============================
# 主程序入口
# =============================
if __name__ == "__main__":
    # 载入数据
    enso_bucket = load_enso() if "ENSO" in control_modes else None
    mjo_bucket = load_mjo() if "MJO" in control_modes else None
    ao_bucket = load_ao() if "AO" in control_modes else None  # 新增: 载入 AO 数据

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

    # 时间对齐和空间掩码
    start_date_input = input("起始日期（YYYY-MM-DD HH:MM:SS）: ")
    end_date_input = input("结束日期（YYYY-MM-DD HH:MM:SS）: ")
    start_date = datetime.strptime(start_date_input, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(end_date_input, "%Y-%m-%d %H:%M:%S")
    start_time_ar = date2num(start_date, ar_time_units, ar_time_calendar)
    end_time_ar = date2num(end_date, ar_time_units, ar_time_calendar)
    start_time_precip = date2num(start_date, precip_time_units, precip_time_calendar)
    end_time_precip = date2num(end_date, precip_time_units, precip_time_calendar)
    ar_idx = (ar_time >= start_time_ar) & (ar_time <= end_time_ar)
    precip_idx = (precip_time >= start_time_precip) & (precip_time <= end_time_precip)
    ar_dates_sel = ar_dates[ar_idx]
    precip_dates_sel = precip_dates[precip_idx]

    unified_units = "hours since 1970-01-01 00:00:00"
    calendar = "proleptic_gregorian"
    ar_time_unified = np.round(date2num(ar_dates_sel, unified_units, calendar), 6)
    precip_time_unified = np.round(date2num(precip_dates_sel, unified_units, calendar), 6)
    common_times = np.intersect1d(ar_time_unified, precip_time_unified)
    if len(common_times) == 0:
        print("错误：在指定的时间范围内，两个数据文件没有共同的时间点。")
        exit()

    ar_indices = np.array(
        [np.where(np.round(date2num(ar_dates, unified_units, calendar), 6) == t)[0][0] for t in common_times])
    precip_indices = np.array(
        [np.where(np.round(date2num(precip_dates, unified_units, calendar), 6) == t)[0][0] for t in common_times])

    lon_min, lon_max = 100, 150
    lat_min, lat_max = 10, 60

    precip_lats_all = precip_ds.variables['latitude'][:]
    precip_lons_all = precip_ds.variables['longitude'][:]
    lat_mask = (precip_lats_all >= lat_min) & (precip_lats_all <= lat_max)
    lon_mask = (precip_lons_all >= lon_min) & (precip_lons_all <= lon_max)
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    # 新增: 检查空间索引是否为空
    if lat_indices.size == 0 or lon_indices.size == 0:
        print(f"错误：在指定的经纬度范围 (Lat: {lat_min}-{lat_max}, Lon: {lon_min}-{lon_max}) 内没有找到降水数据格点。")
        print("请检查 precip_ds 文件中的坐标或调整代码中的 lon_min, lon_max, lat_min, lat_max 参数。")
        exit()

    precip_lats = precip_lats_all[lat_indices]
    precip_lons = precip_lons_all[lon_indices]

    ar_lats_all = ar_ds.variables['lat'][:]
    ar_lons_all = ar_ds.variables['lon'][:]
    ar_lat_mask = (ar_lats_all >= lat_min) & (ar_lats_all <= lat_max)
    ar_lon_mask = (ar_lons_all >= lon_min) & (ar_lons_all <= lon_max)
    ar_lat_indices = np.where(ar_lat_mask)[0]
    ar_lon_indices = np.where(ar_lon_mask)[0]

    # 新增: 检查空间索引是否为空
    if ar_lat_indices.size == 0 or ar_lon_indices.size == 0:
        print(f"错误：在指定的经纬度范围 (Lat: {lat_min}-{lat_max}, Lon: {lon_min}-{lon_max}) 内没有找到 AR 数据格点。")
        print("请检查 ar_happen.nc 文件中的坐标或调整代码中的 lon_min, lon_max, lat_min, lat_max 参数。")
        exit()

    ar_lats = ar_lats_all[ar_lat_indices]
    ar_lons = ar_lons_all[ar_lon_indices]

    if ar_lats[0] > ar_lats[-1]:
        ar_lats = ar_lats[::-1]
        ar_lat_indices = ar_lat_indices[::-1]
    if precip_lats[0] > precip_lats[-1]:
        precip_lats = precip_lats[::-1]
        lat_indices = lat_indices[::-1]

    ar_data = ar_ds.variables['ar_happen'][ar_indices, :, :][:, ar_lat_indices, :][:, :, ar_lon_indices].astype(np.int8)

    if 'extreme_precipitation_flag_ocean' in precip_ds.variables and 'extreme_precipitation_flag_land' in precip_ds.variables:
        precip_ocean = precip_ds.variables['extreme_precipitation_flag_ocean'][precip_indices, :, :][:, lat_indices, :][
                       :, :, lon_indices]
        precip_land = precip_ds.variables['extreme_precipitation_flag_land'][precip_indices, :, :][:, lat_indices, :][:,
                      :, lon_indices]
        precip_data = np.maximum(precip_ocean, precip_land).astype(np.int8)
    else:
        precip_data = precip_ds.variables['extreme_precipitation_flag'][precip_indices, :, :][:, lat_indices, :][:, :,
                      lon_indices].astype(np.int8)

    # 运行逻辑回归分析
    if any(mode in control_modes for mode in ["ENSO", "MJO", "AO"]):
        run_logistic_regression(ar_data, precip_data, ar_dates_sel, enso_bucket, mjo_bucket, ao_bucket)

    # 运行区域异质性分析
    run_regional_analysis(ar_data, precip_data, ar_dates_sel, ar_lats, ar_lons)

    ar_ds.close()
    precip_ds.close()
    print("\n所有计算完成，结果已保存。")
