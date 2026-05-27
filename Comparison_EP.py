import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 构造样例数据
# =========================

np.random.seed(42)

n_years = 30
days_per_year = 365
omega = 15
q = 95

years = np.arange(n_years)
days = np.arange(1, days_per_year + 1)

records = []

for year in years:
    for d in days:
        # 季节性背景：夏季降水更强，冬季更弱
        seasonal_scale = 5 + 18 * np.exp(-((d - 200) / 60) ** 2)

        # 提高湿日概率，使灰色样本更密
        wet_prob = 0.45 + 0.35 * np.exp(-((d - 200) / 75) ** 2)

        if np.random.rand() < wet_prob:
            precip = np.random.gamma(shape=2.0, scale=seasonal_scale)
        else:
            precip = 0.0

        records.append([year, d, precip])

df = pd.DataFrame(records, columns=["year", "day", "precip"])

# =========================
# 2. 给样例年份增加局部峰值
# =========================

example_year = 0

def add_local_peak(df, year, center_day, amplitude, width):
    """
    在指定年份、指定日期附近加入一个高斯形状的局部峰值。
    只修改同一个样例年份，不改变样本空间。
    """
    mask = df["year"] == year
    d = df.loc[mask, "day"].values

    peak = amplitude * np.exp(-((d - center_day) / width) ** 2)

    df.loc[mask, "precip"] += peak

    return df

# 增加若干局部峰值，但不在图中额外标注
df = add_local_peak(df, example_year, center_day=45, amplitude=28, width=3)
df = add_local_peak(df, example_year, center_day=95, amplitude=22, width=4)
df = add_local_peak(df, example_year, center_day=130, amplitude=36, width=4)
df = add_local_peak(df, example_year, center_day=205, amplitude=45, width=5)
df = add_local_peak(df, example_year, center_day=240, amplitude=35, width=4)
df = add_local_peak(df, example_year, center_day=305, amplitude=30, width=4)

# =========================
# 3. 方法 A：全局固定阈值
# =========================

wet_samples_global = df.loc[df["precip"] > 0, "precip"]
global_threshold = np.percentile(wet_samples_global, q)

df["global_threshold"] = global_threshold
df["extreme_global"] = df["precip"] > df["global_threshold"]

# =========================
# 4. 方法 B：历日滑动窗口相对阈值
# =========================

daily_thresholds = []

for d in days:
    window_days = []

    for offset in range(-omega, omega + 1):
        dd = d + offset

        if dd < 1:
            dd += days_per_year
        elif dd > days_per_year:
            dd -= days_per_year

        window_days.append(dd)

    samples = df.loc[
        (df["day"].isin(window_days)) & (df["precip"] > 0),
        "precip"
    ]

    threshold = np.percentile(samples, q)
    daily_thresholds.append(threshold)

threshold_df = pd.DataFrame({
    "day": days,
    "moving_threshold": daily_thresholds
})

df = df.merge(threshold_df, on="day", how="left")
df["extreme_moving"] = df["precip"] > df["moving_threshold"]

df_one_year = df[df["year"] == example_year].copy()

# =========================
# 5. 绘图比较
# =========================

plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# ---------- 图1：样例年份降水序列 ----------
axes[0].bar(
    df_one_year["day"],
    df_one_year["precip"],
    color="lightgray",
    edgecolor="none",
    label="Daily precipitation"
)

axes[0].axhline(
    global_threshold,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Global fixed 95th percentile"
)

axes[0].plot(
    days,
    daily_thresholds,
    color="blue",
    linewidth=2,
    label="Calendar-day moving-window 95th percentile"
)

axes[0].set_ylabel("Precipitation")
axes[0].set_title("Comparison of Two Extreme Precipitation Threshold Definitions")
axes[0].legend()
axes[0].grid(alpha=0.3)

# ---------- 图2：全局固定阈值识别出的极端事件 ----------
axes[1].bar(
    df_one_year["day"],
    df_one_year["precip"],
    color="lightgray",
    edgecolor="none",
    label="Non-extreme precipitation"
)

extreme_global = df_one_year[df_one_year["extreme_global"]]

axes[1].bar(
    extreme_global["day"],
    extreme_global["precip"],
    color="red",
    edgecolor="none",
    label="Extreme by global fixed threshold"
)

axes[1].axhline(
    global_threshold,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Global fixed threshold"
)

axes[1].set_ylabel("Precipitation")
axes[1].set_title("Extreme Events Detected by Global Fixed Threshold")
axes[1].legend()
axes[1].grid(alpha=0.3)

# ---------- 图3：滑动窗口相对阈值识别出的极端事件 ----------
axes[2].bar(
    df_one_year["day"],
    df_one_year["precip"],
    color="lightgray",
    edgecolor="none",
    label="Non-extreme precipitation"
)

extreme_moving = df_one_year[df_one_year["extreme_moving"]]

axes[2].bar(
    extreme_moving["day"],
    extreme_moving["precip"],
    color="blue",
    edgecolor="none",
    label="Extreme by moving-window threshold"
)

axes[2].plot(
    days,
    daily_thresholds,
    color="blue",
    linewidth=2,
    label="Moving-window threshold"
)

axes[2].set_xlabel("Day of Year")
axes[2].set_ylabel("Precipitation")
axes[2].set_title("Extreme Events Detected by Calendar-Day Moving-Window Threshold")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =========================
# 6. 简单输出两种方法识别出的极端事件数量
# =========================

print("Global fixed threshold:", round(global_threshold, 2))
print(
    "Number of extremes by global threshold in example year:",
    df_one_year["extreme_global"].sum()
)
print(
    "Number of extremes by moving-window threshold in example year:",
    df_one_year["extreme_moving"].sum()
)
print(
    "Overlap extremes:",
    (df_one_year["extreme_global"] & df_one_year["extreme_moving"]).sum()
)
print(
    "Only global extremes:",
    (df_one_year["extreme_global"] & ~df_one_year["extreme_moving"]).sum()
)
print(
    "Only moving-window extremes:",
    (~df_one_year["extreme_global"] & df_one_year["extreme_moving"]).sum()
)
