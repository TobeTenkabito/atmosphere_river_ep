import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# 0. 基本设置
# ============================================================

np.random.seed(2027)
plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# 1. 构造示意样本空间
# ============================================================
# 每个样本代表一个 grid-time 单元。
# 横轴：月份
# 纵轴：格点
# 每个样本有：
#   month: 日历月份
#   grid: 空间格点
#   region: 区域背景
#   AR: 前期 AR 暴露
#   EP: 极端降水事件

n_points = 6000

months = np.random.choice(np.arange(1, 13), size=n_points)
grids = np.random.choice(np.arange(1, 11), size=n_points)

# 区域划分：低编号格点为内陆，高编号格点为沿海湿润区
regions = np.where(
    grids <= 3,
    "Dry inland",
    np.where(grids <= 7, "Transition", "Coastal humid")
)

region_code = np.where(
    regions == "Dry inland",
    0,
    np.where(regions == "Transition", 1, 2)
)

# 季节背景
is_summer = np.isin(months, [6, 7, 8])
is_coastal = regions == "Coastal humid"

# ------------------------------------------------------------
# 构造 AR 暴露
# ------------------------------------------------------------
# 让 AR 更容易发生在夏季和沿海区域。
# 这会使传统 AR / non-AR 比较天然存在季节和空间组成差异。
p_ar = (
    0.08
    + 0.33 * is_summer.astype(float)
    + 0.25 * is_coastal.astype(float)
)

p_ar = np.clip(p_ar, 0.02, 0.85)
AR = np.random.rand(n_points) < p_ar

# ------------------------------------------------------------
# 构造 EP 事件
# ------------------------------------------------------------
# EP 同时受三个因素影响：
#   1. AR 暴露
#   2. 夏季背景
#   3. 沿海湿润背景
#
# 这样传统方法会把 AR 影响、季节影响和空间影响混在一起。
p_ep = (
    0.015
    + 0.090 * AR.astype(float)
    + 0.060 * is_summer.astype(float)
    + 0.055 * is_coastal.astype(float)
)

p_ep = np.clip(p_ep, 0.005, 0.65)
EP = np.random.rand(n_points) < p_ep

# ============================================================
# 2. 构造时空匹配事件-对照样本
# ============================================================
# 对每一个 EP 事件：
# 从同一个 grid、同一个 month 中抽取若干个非 EP 样本作为对照。
# 这样事件和对照在 grid 和 month 上完全匹配。

n_controls = 5

event_idx_all = np.where(EP)[0]

matched_events = []
matched_controls = []
matched_group_id = []

group_id = 0

for idx in event_idx_all:
    g = grids[idx]
    m = months[idx]

    candidates = np.where(
        (grids == g) &
        (months == m) &
        (~EP)
    )[0]

    if len(candidates) >= n_controls:
        chosen = np.random.choice(
            candidates,
            size=n_controls,
            replace=False
        )

        matched_events.append(idx)
        matched_controls.extend(chosen)

        matched_group_id.extend([group_id] * (1 + n_controls))
        group_id += 1

matched_events = np.array(matched_events)
matched_controls = np.array(matched_controls)

print(f"Number of matched event groups: {len(matched_events)}")
print(f"Number of matched controls: {len(matched_controls)}")

# ============================================================
# 3. 传统方法估计
# ============================================================

p_ep_non_ar = EP[~AR].mean()
p_ep_ar = EP[AR].mean()
traditional_rr = p_ep_ar / p_ep_non_ar

# ============================================================
# 4. 匹配方法估计
# ============================================================

event_ar_rate = AR[matched_events].mean()
control_ar_rate = AR[matched_controls].mean()

# 近似 OR，加 0.5 防止极端情况下分母为 0
a = AR[matched_events].sum() + 0.5
b = (~AR[matched_events]).sum() + 0.5
c = AR[matched_controls].sum() + 0.5
d = (~AR[matched_controls]).sum() + 0.5

matched_or = (a / b) / (c / d)

# ============================================================
# 5. 计算组成不平衡度
# ============================================================
# 使用 Total Variation Distance:
#
# imbalance = 0.5 * sum_k |p1_k - p0_k|
#
# 0 表示两组组成完全一致；
# 越大表示两组组成越不平衡。

def composition(indices, category_array, n_category):
    out = []
    for k in range(n_category):
        out.append(np.mean(category_array[indices] == k))
    return np.array(out)


def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))


# 月份分组：四季尺度
# 0: Jan-Mar
# 1: Apr-Jun
# 2: Jul-Sep
# 3: Oct-Dec
month_group = np.zeros_like(months)

month_group[np.isin(months, [1, 2, 3])] = 0
month_group[np.isin(months, [4, 5, 6])] = 1
month_group[np.isin(months, [7, 8, 9])] = 2
month_group[np.isin(months, [10, 11, 12])] = 3

idx_non_ar = np.where(~AR)[0]
idx_ar = np.where(AR)[0]

# 传统 AR vs non-AR 的季节组成
trad_season_non_ar = composition(idx_non_ar, month_group, 4)
trad_season_ar = composition(idx_ar, month_group, 4)

# 匹配 event vs control 的季节组成
match_season_event = composition(matched_events, month_group, 4)
match_season_control = composition(matched_controls, month_group, 4)

# 传统 AR vs non-AR 的空间组成
trad_region_non_ar = composition(idx_non_ar, region_code, 3)
trad_region_ar = composition(idx_ar, region_code, 3)

# 匹配 event vs control 的空间组成
match_region_event = composition(matched_events, region_code, 3)
match_region_control = composition(matched_controls, region_code, 3)

# 不平衡度
season_imbalance_trad = total_variation_distance(
    trad_season_non_ar,
    trad_season_ar
)

season_imbalance_match = total_variation_distance(
    match_season_control,
    match_season_event
)

spatial_imbalance_trad = total_variation_distance(
    trad_region_non_ar,
    trad_region_ar
)

spatial_imbalance_match = total_variation_distance(
    match_region_control,
    match_region_event
)

# ============================================================
# 6. 扰动检验
# ============================================================
# 这里不是强行抬高 observed OR。
# 而是在 matched 样本内部随机打乱 AR 暴露状态，模拟破坏真实 AR-EP 对应关系。
# 然后重新计算 OR。

def compute_or(event_ar_vec, control_ar_vec):
    a = event_ar_vec.sum() + 0.5
    b = (~event_ar_vec).sum() + 0.5
    c = control_ar_vec.sum() + 0.5
    d = (~control_ar_vec).sum() + 0.5
    return (a / b) / (c / d)


n_perm = 500
perturbed_ors = []

combined_ar = np.concatenate([
    AR[matched_events],
    AR[matched_controls]
])

n_event = len(matched_events)

for _ in range(n_perm):
    shuffled = np.random.permutation(combined_ar)

    event_ar_perm = shuffled[:n_event]
    control_ar_perm = shuffled[n_event:]

    perturbed_ors.append(
        compute_or(event_ar_perm, control_ar_perm)
    )

perturbed_ors = np.array(perturbed_ors)
perturb_p95 = np.percentile(perturbed_ors, 95)

# ============================================================
# Figure 1: Clean schematic of sample selection
# ============================================================
# 极简版：
#   (a) 原始样本空间
#   (b) 传统方法：两堆样本
#   (c) 匹配方法：同一 grid-month 内的事件-对照比较

from matplotlib.lines import Line2D

rng_fig1 = np.random.default_rng(2030)

# ------------------------------------------------------------
# 1. 可视化抖动
# ------------------------------------------------------------

day_like = rng_fig1.integers(1, 31, size=n_points)
x_jittered = months + (day_like - 15.5) * 0.010
y_jittered = grids + rng_fig1.normal(0, 0.030, size=n_points)

# ------------------------------------------------------------
# 2. 选择一个展示用 matched event
# ------------------------------------------------------------

if len(matched_events) == 0:
    raise ValueError("No matched events found. Please increase n_points or lower n_controls.")

matched_events_with_ar = matched_events[AR[matched_events]]

if len(matched_events_with_ar) > 0:
    chosen_event = matched_events_with_ar[0]
else:
    chosen_event = matched_events[0]

chosen_event_pos = np.where(matched_events == chosen_event)[0][0]

control_start = chosen_event_pos * n_controls
control_end = control_start + n_controls
chosen_controls = matched_controls[control_start:control_end]

event_grid = grids[chosen_event]
event_month = months[chosen_event]

same_stratum_idx = np.where(
    (grids == event_grid) &
    (months == event_month)
)[0]

same_stratum_non_event = same_stratum_idx[~EP[same_stratum_idx]]

other_same_stratum = np.setdiff1d(
    same_stratum_non_event,
    chosen_controls
)

n_other_show = min(8, len(other_same_stratum))

if n_other_show > 0:
    other_show = rng_fig1.choice(
        other_same_stratum,
        size=n_other_show,
        replace=False
    )
else:
    other_show = np.array([], dtype=int)

local_show_ordered = np.concatenate([
    other_show[:4],
    chosen_controls,
    other_show[4:],
    [chosen_event]
])

# ------------------------------------------------------------
# 3. 创建画布
# ------------------------------------------------------------

fig = plt.figure(figsize=(13.5, 8.0))

gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[1.0, 0.85],
    width_ratios=[1.08, 1.0],
    hspace=0.42,
    wspace=0.32
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])

# ============================================================
# Panel A: 原始样本空间
# ============================================================

ax = ax_a

plot_idx_fig1 = rng_fig1.choice(
    np.arange(n_points),
    size=min(800, n_points),
    replace=False
)

plot_idx_fig1 = np.unique(
    np.concatenate([
        plot_idx_fig1,
        [chosen_event],
        chosen_controls
    ])
)

idx_non = plot_idx_fig1[~AR[plot_idx_fig1]]
idx_ar = plot_idx_fig1[AR[plot_idx_fig1]]
event_plot_idx = plot_idx_fig1[EP[plot_idx_fig1]]

ax.scatter(
    x_jittered[idx_non],
    y_jittered[idx_non],
    s=24,
    c="#D0D0D0",
    edgecolors="white",
    linewidths=0.3,
    alpha=0.70
)

ax.scatter(
    x_jittered[idx_ar],
    y_jittered[idx_ar],
    s=30,
    c="#D62728",
    edgecolors="white",
    linewidths=0.3,
    alpha=0.85
)

ax.scatter(
    x_jittered[event_plot_idx],
    y_jittered[event_plot_idx],
    s=72,
    facecolors="none",
    edgecolors="black",
    linewidths=0.95
)

ax.scatter(
    x_jittered[chosen_controls],
    y_jittered[chosen_controls],
    s=76,
    c="#4E79A7",
    edgecolors="white",
    linewidths=0.7,
    zorder=5
)

ax.scatter(
    x_jittered[chosen_event],
    y_jittered[chosen_event],
    s=145,
    marker="*",
    c="#F28E2B",
    edgecolors="black",
    linewidths=0.8,
    zorder=6
)

rect = patches.Rectangle(
    (event_month - 0.48, event_grid - 0.38),
    0.96,
    0.76,
    linewidth=1.3,
    edgecolor="#F28E2B",
    facecolor="none",
    linestyle="--"
)
ax.add_patch(rect)

ax.text(
    event_month,
    event_grid + 0.58,
    "selected stratum",
    ha="center",
    va="bottom",
    fontsize=8,
    color="#F28E2B"
)

ax.set_title("(a) Original samples", fontsize=12)
ax.set_xlabel("Month")
ax.set_ylabel("Grid")
ax.set_xticks(np.arange(1, 13))
ax.set_yticks(np.arange(1, 11))
ax.set_xlim(0.5, 12.5)
ax.set_ylim(0.5, 10.5)

# ============================================================
# Panel B: 传统方法，两堆样本
# ============================================================

ax = ax_b

non_ar_idx = np.where(~AR)[0]
ar_idx = np.where(AR)[0]

max_show = 75

non_ar_show = rng_fig1.choice(
    non_ar_idx,
    size=min(max_show, len(non_ar_idx)),
    replace=False
)

ar_show = rng_fig1.choice(
    ar_idx,
    size=min(max_show, len(ar_idx)),
    replace=False
)

# 加一些 EP 样本
non_ar_ep = non_ar_idx[EP[non_ar_idx]]
ar_ep = ar_idx[EP[ar_idx]]

if len(non_ar_ep) > 0:
    non_ar_show = np.unique(
        np.concatenate([
            non_ar_show,
            rng_fig1.choice(non_ar_ep, size=min(8, len(non_ar_ep)), replace=False)
        ])
    )

if len(ar_ep) > 0:
    ar_show = np.unique(
        np.concatenate([
            ar_show,
            rng_fig1.choice(ar_ep, size=min(8, len(ar_ep)), replace=False)
        ])
    )

x_non = 0 + rng_fig1.normal(0, 0.075, size=len(non_ar_show))
y_non = rng_fig1.uniform(0.14, 0.82, size=len(non_ar_show))

x_ar = 1 + rng_fig1.normal(0, 0.075, size=len(ar_show))
y_ar = rng_fig1.uniform(0.14, 0.82, size=len(ar_show))

ax.scatter(
    x_non,
    y_non,
    s=32,
    c="#D0D0D0",
    edgecolors="white",
    linewidths=0.3,
    alpha=0.85
)

ax.scatter(
    x_ar,
    y_ar,
    s=36,
    c="#D62728",
    edgecolors="white",
    linewidths=0.3,
    alpha=0.90
)

ep_non = EP[non_ar_show]
ep_ar = EP[ar_show]

ax.scatter(
    x_non[ep_non],
    y_non[ep_non],
    s=78,
    facecolors="none",
    edgecolors="black",
    linewidths=0.95
)

ax.scatter(
    x_ar[ep_ar],
    y_ar[ep_ar],
    s=78,
    facecolors="none",
    edgecolors="black",
    linewidths=0.95
)

for xpos, label in zip([0, 1], ["Non-AR", "AR"]):
    rect = patches.Rectangle(
        (xpos - 0.28, 0.08),
        0.56,
        0.82,
        linewidth=1.25,
        edgecolor="#555555",
        facecolor="none"
    )
    ax.add_patch(rect)

    ax.text(
        xpos,
        0.94,
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

ax.set_title("(b) Conventional: pooled comparison", fontsize=12)
ax.set_xlim(-0.55, 1.55)
ax.set_ylim(0.0, 1.05)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Non-AR days", "AR days"])
ax.set_yticks([])

# 简短说明，放在图内底部
ax.text(
    0.5,
    0.02,
    "All grids and months pooled",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#555555",
    transform=ax.transAxes
)

# ============================================================
# Panel C: 匹配方法
# ============================================================

ax = ax_c

positions = np.arange(1, len(local_show_ordered) + 1)

ax.hlines(
    y=0.55,
    xmin=0.5,
    xmax=len(local_show_ordered) + 0.5,
    color="#D0D0D0",
    linewidth=2.0,
    zorder=1
)

for pos, idx in zip(positions, local_show_ordered):

    if idx == chosen_event:
        color = "#F28E2B"
        marker = "*"
        size = 220
        edgecolor = "black"
        lw = 1.0
        z = 5
    elif idx in chosen_controls:
        color = "#4E79A7"
        marker = "o"
        size = 115
        edgecolor = "white"
        lw = 0.8
        z = 4
    else:
        color = "#D62728" if AR[idx] else "#D0D0D0"
        marker = "o"
        size = 85
        edgecolor = "black" if EP[idx] else "white"
        lw = 1.0 if EP[idx] else 0.6
        z = 3

    ax.scatter(
        pos,
        0.55,
        s=size,
        c=color,
        marker=marker,
        edgecolors=edgecolor,
        linewidths=lw,
        zorder=z
    )

# 对照框
control_positions = []

for idx in chosen_controls:
    pos = np.where(local_show_ordered == idx)[0][0] + 1
    control_positions.append(pos)

    rect = patches.Rectangle(
        (pos - 0.35, 0.42),
        0.70,
        0.26,
        linewidth=1.25,
        edgecolor="#4E79A7",
        facecolor="none",
        linestyle="--"
    )
    ax.add_patch(rect)

# 事件框
event_pos = np.where(local_show_ordered == chosen_event)[0][0] + 1

rect = patches.Rectangle(
    (event_pos - 0.42, 0.38),
    0.84,
    0.34,
    linewidth=1.55,
    edgecolor="#F28E2B",
    facecolor="none",
    linestyle="-"
)
ax.add_patch(rect)

if len(control_positions) > 0:
    ax.text(
        np.mean(control_positions),
        0.83,
        "Controls",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4E79A7",
        fontweight="bold"
    )

ax.text(
    event_pos,
    0.83,
    "Event",
    ha="center",
    va="bottom",
    fontsize=10,
    color="#F28E2B",
    fontweight="bold"
)

ax.text(
    0.02,
    0.92,
    f"Same grid-month: grid {event_grid}, month {event_month}",
    transform=ax.transAxes,
    ha="left",
    va="center",
    fontsize=11,
    fontweight="bold"
)

ax.set_title("(c) Matched: local event-control comparison", fontsize=12)
ax.set_xlim(0.4, len(local_show_ordered) + 0.6)
ax.set_ylim(0.15, 1.02)
ax.set_xticks([])
ax.set_yticks([])

# ============================================================
# 统一图例
# ============================================================

legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Non-AR day",
        markerfacecolor="#D0D0D0",
        markeredgecolor="white",
        markersize=7
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="AR day",
        markerfacecolor="#D62728",
        markeredgecolor="white",
        markersize=7
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Extreme precipitation",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=8
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Matched control",
        markerfacecolor="#4E79A7",
        markeredgecolor="white",
        markersize=7
    ),
    Line2D(
        [0], [0],
        marker="*",
        color="w",
        label="Selected event",
        markerfacecolor="#F28E2B",
        markeredgecolor="black",
        markersize=11
    )
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=5,
    frameon=True,
    fontsize=9,
    bbox_to_anchor=(0.5, 0.02)
)

fig.suptitle(
    "Sample Selection in Conventional and Matched Designs",
    fontsize=15,
    y=0.97
)

plt.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.90,
    bottom=0.12
)

plt.show()



# ============================================================
# Figure 2: 新方法优越性核心图 —— 不平衡度下降
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.8))

labels = ["Seasonal\nimbalance", "Spatial\nimbalance"]
x = np.arange(len(labels))
width = 0.34

traditional_values = [
    season_imbalance_trad,
    spatial_imbalance_trad
]

matched_values = [
    season_imbalance_match,
    spatial_imbalance_match
]

bars1 = ax.bar(
    x - width / 2,
    traditional_values,
    width,
    color="#D62728",
    edgecolor="black",
    linewidth=0.8,
    label="Conventional AR / non-AR comparison"
)

bars2 = ax.bar(
    x + width / 2,
    matched_values,
    width,
    color="#4E79A7",
    edgecolor="black",
    linewidth=0.8,
    label="Matched event-control design"
)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Imbalance score", fontsize=12)
ax.set_title(
    "Matching Substantially Reduces Seasonal and Spatial Imbalance",
    fontsize=14
)

ymax = max(max(traditional_values), max(matched_values)) * 1.35
ax.set_ylim(0, ymax)

ax.legend(frameon=True, fontsize=10)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + ymax * 0.025,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

ax.text(
    0.5,
    -0.18,
    "Lower values indicate more comparable groups. Matching makes events and controls nearly identical in grid-month background.",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=10
)

plt.tight_layout()
plt.show()

# ============================================================
# Figure 3: 组成图，辅助说明不平衡来自哪里
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

season_labels = ["Jan-Mar", "Apr-Jun", "Jul-Sep", "Oct-Dec"]
season_colors = ["#DDEBF7", "#B7DEE8", "#FDBE85", "#B2DF8A"]

region_labels = ["Dry inland", "Transition", "Coastal humid"]
region_colors = ["#E5E5E5", "#A6CEE3", "#1F78B4"]

# ------------------------------------------------------------
# 3a 传统季节组成
# ------------------------------------------------------------

ax = axes[0, 0]
bottom = np.zeros(2)
data = np.vstack([trad_season_non_ar, trad_season_ar])

for k in range(4):
    ax.bar(
        ["Non-AR", "AR"],
        data[:, k],
        bottom=bottom,
        color=season_colors[k],
        edgecolor="white",
        label=season_labels[k]
    )
    bottom += data[:, k]

ax.set_ylim(0, 1)
ax.set_ylabel("Composition")
ax.set_title("(a) Conventional: seasonal composition differs")
ax.legend(frameon=True, fontsize=8, loc="upper right")

# ------------------------------------------------------------
# 3b 匹配季节组成
# ------------------------------------------------------------

ax = axes[0, 1]
bottom = np.zeros(2)
data = np.vstack([match_season_control, match_season_event])

for k in range(4):
    ax.bar(
        ["Matched controls", "Events"],
        data[:, k],
        bottom=bottom,
        color=season_colors[k],
        edgecolor="white",
        label=season_labels[k]
    )
    bottom += data[:, k]

ax.set_ylim(0, 1)
ax.set_ylabel("Composition")
ax.set_title("(b) Matched: seasonal composition is balanced")
ax.legend(frameon=True, fontsize=8, loc="upper right")

# ------------------------------------------------------------
# 3c 传统空间组成
# ------------------------------------------------------------

ax = axes[1, 0]
bottom = np.zeros(2)
data = np.vstack([trad_region_non_ar, trad_region_ar])

for k in range(3):
    ax.bar(
        ["Non-AR", "AR"],
        data[:, k],
        bottom=bottom,
        color=region_colors[k],
        edgecolor="white",
        label=region_labels[k]
    )
    bottom += data[:, k]

ax.set_ylim(0, 1)
ax.set_ylabel("Composition")
ax.set_title("(c) Conventional: spatial composition differs")
ax.legend(frameon=True, fontsize=8, loc="upper right")

# ------------------------------------------------------------
# 3d 匹配空间组成
# ------------------------------------------------------------

ax = axes[1, 1]
bottom = np.zeros(2)
data = np.vstack([match_region_control, match_region_event])

for k in range(3):
    ax.bar(
        ["Matched controls", "Events"],
        data[:, k],
        bottom=bottom,
        color=region_colors[k],
        edgecolor="white",
        label=region_labels[k]
    )
    bottom += data[:, k]

ax.set_ylim(0, 1)
ax.set_ylabel("Composition")
ax.set_title("(d) Matched: spatial composition is balanced")
ax.legend(frameon=True, fontsize=8, loc="upper right")

plt.suptitle(
    "Figure 3. Matching Balances the Background Composition of Events and Controls",
    fontsize=15,
    y=1.02
)

plt.tight_layout()
plt.show()

# ============================================================
# Figure 4: 估计结果与扰动检验
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# ------------------------------------------------------------
# 4a 传统方法估计
# ------------------------------------------------------------

ax = axes[0]

bars = ax.bar(
    ["Non-AR", "AR"],
    [p_ep_non_ar, p_ep_ar],
    color=["#BDBDBD", "#D62728"],
    edgecolor="black",
    width=0.62
)

ax.set_ylabel("Probability of extreme precipitation")
ax.set_title("(a) Conventional estimate\npotentially affected by background imbalance")

ax.text(
    0.5,
    max(p_ep_non_ar, p_ep_ar) * 1.12,
    f"Risk ratio = {traditional_rr:.2f}",
    ha="center",
    fontsize=11
)

ax.set_ylim(0, max(p_ep_non_ar, p_ep_ar) * 1.35)

for bar in bars:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.004,
        f"{h:.3f}",
        ha="center",
        fontsize=10
    )

# ------------------------------------------------------------
# 4b 匹配方法估计
# ------------------------------------------------------------

ax = axes[1]

bars = ax.bar(
    ["Matched controls", "Events"],
    [control_ar_rate, event_ar_rate],
    color=["#4E79A7", "#F28E2B"],
    edgecolor="black",
    width=0.62
)

ax.set_ylabel("Probability of prior AR exposure")
ax.set_title("(b) Matched event-control estimate\nunder balanced grid-month background")

ax.text(
    0.5,
    max(control_ar_rate, event_ar_rate) * 1.12,
    f"Odds ratio ≈ {matched_or:.2f}",
    ha="center",
    fontsize=11
)

ax.set_ylim(0, max(control_ar_rate, event_ar_rate) * 1.35)

for bar in bars:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.025,
        f"{h:.2f}",
        ha="center",
        fontsize=10
    )

# ------------------------------------------------------------
# 4c 扰动检验
# ------------------------------------------------------------

ax = axes[2]

ax.hist(
    perturbed_ors,
    bins=30,
    color="#BDBDBD",
    edgecolor="white",
    alpha=0.95,
    label="Perturbed AR exposure"
)

ax.axvline(
    perturb_p95,
    color="#666666",
    linestyle="--",
    linewidth=2.2,
    label="95th percentile"
)

ax.axvline(
    matched_or,
    color="#D62728",
    linewidth=3,
    label="Observed matched estimate"
)

ax.set_xlabel("Estimated odds ratio")
ax.set_ylabel("Frequency")
ax.set_title("(c) Perturbation test\nobserved association vs disrupted timing")
ax.legend(frameon=True, fontsize=9)

plt.suptitle(
    "Figure 4. Association Estimates and Robustness Check",
    fontsize=15,
    y=1.05
)

plt.tight_layout()
plt.show()

# ============================================================
# 结果输出
# ============================================================

print("\n==============================")
print("Traditional AR / non-AR comparison")
print("==============================")
print(f"P(EP | Non-AR) = {p_ep_non_ar:.4f}")
print(f"P(EP | AR)     = {p_ep_ar:.4f}")
print(f"Risk ratio     = {traditional_rr:.2f}")

print("\n==============================")
print("Matched event-control comparison")
print("==============================")
print(f"P(AR | matched controls) = {control_ar_rate:.4f}")
print(f"P(AR | events)           = {event_ar_rate:.4f}")
print(f"Matched OR               = {matched_or:.2f}")

print("\n==============================")
print("Imbalance reduction")
print("==============================")
print(f"Seasonal imbalance, conventional = {season_imbalance_trad:.3f}")
print(f"Seasonal imbalance, matched      = {season_imbalance_match:.3f}")
print(f"Spatial imbalance, conventional  = {spatial_imbalance_trad:.3f}")
print(f"Spatial imbalance, matched       = {spatial_imbalance_match:.3f}")

print("\n==============================")
print("Perturbation test")
print("==============================")
print(f"Observed matched OR              = {matched_or:.2f}")
print(f"95th percentile of perturbations = {perturb_p95:.2f}")
print(f"Observed > 95% perturbations     = {matched_or > perturb_p95}")
