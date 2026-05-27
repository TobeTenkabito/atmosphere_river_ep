import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl


# ============================================================
# Clean ENSO/MJO diagnostic workflow schematic
# ============================================================

def set_clean_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white"
    })


def add_card(ax, x, y, w, h, step, title, body, color):
    """
    Card layout:
    top    : step + title
    middle : body text
    bottom : icon area
    """
    # shadow
    shadow = patches.FancyBboxPatch(
        (x + 0.006, y - 0.006),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        linewidth=0,
        facecolor="#D9D9D9",
        alpha=0.30,
        zorder=1
    )
    ax.add_patch(shadow)

    # main card
    card = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        linewidth=1.1,
        edgecolor="#333333",
        facecolor="white",
        zorder=2
    )
    ax.add_patch(card)

    # top color band
    band_h = h * 0.24

    band = patches.FancyBboxPatch(
        (x, y + h - band_h),
        w,
        band_h,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        linewidth=0,
        facecolor=color,
        zorder=3
    )
    ax.add_patch(band)

    # mask lower rounded part of band
    ax.add_patch(
        patches.Rectangle(
            (x, y + h - band_h),
            w,
            band_h * 0.30,
            linewidth=0,
            facecolor=color,
            zorder=3
        )
    )

    # step circle
    circle = patches.Circle(
        (x + 0.035, y + h - band_h / 2),
        radius=0.024,
        facecolor="white",
        edgecolor="#333333",
        linewidth=1.0,
        zorder=4
    )
    ax.add_patch(circle)

    ax.text(
        x + 0.035,
        y + h - band_h / 2,
        str(step),
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#222222",
        zorder=5
    )

    # title
    ax.text(
        x + w * 0.58,
        y + h - band_h / 2,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#111111",
        zorder=5
    )

    # body text: middle zone
    ax.text(
        x + w / 2,
        y + h * 0.50,
        body,
        ha="center",
        va="center",
        fontsize=8.4,
        color="#333333",
        linespacing=1.35,
        zorder=5
    )


def add_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.5,
            color="#555555",
            shrinkA=5,
            shrinkB=5
        ),
        zorder=10
    )


def draw_input_icon(ax, cx, cy, scale=1.0):
    """
    Icon for gridded AR/EP and climate indices.
    """
    # grid dots
    for i in range(3):
        for j in range(3):
            ax.scatter(
                cx - 0.026 * scale + j * 0.026 * scale,
                cy - 0.014 * scale + i * 0.014 * scale,
                s=13 * scale,
                color="#D0D0D0",
                edgecolor="white",
                linewidth=0.35,
                zorder=6
            )

    # AR red point
    ax.scatter(
        cx,
        cy + 0.014 * scale,
        s=26 * scale,
        color="#D62728",
        edgecolor="white",
        linewidth=0.5,
        zorder=7
    )

    # EP circle
    ax.scatter(
        cx + 0.026 * scale,
        cy,
        s=38 * scale,
        facecolors="none",
        edgecolors="black",
        linewidth=0.8,
        zorder=7
    )

    # climate index line
    xs = [
        cx - 0.045 * scale,
        cx - 0.022 * scale,
        cx,
        cx + 0.022 * scale,
        cx + 0.045 * scale
    ]

    ys = [
        cy - 0.050 * scale,
        cy - 0.039 * scale,
        cy - 0.055 * scale,
        cy - 0.038 * scale,
        cy - 0.048 * scale
    ]

    ax.plot(xs, ys, color="#4E79A7", lw=1.2, zorder=7)


def draw_lag_icon(ax, cx, cy, scale=1.0):
    """
    Icon for lagged alignment.
    """
    ax.plot(
        [cx - 0.058 * scale, cx + 0.058 * scale],
        [cy, cy],
        color="#4E79A7",
        lw=1.5,
        zorder=6
    )

    tick_x = [
        cx - 0.058 * scale,
        cx,
        cx + 0.058 * scale
    ]

    tick_labels = ["0", "lag", "max"]

    for xx, lab in zip(tick_x, tick_labels):
        ax.plot(
            [xx, xx],
            [cy - 0.010 * scale, cy + 0.010 * scale],
            color="#4E79A7",
            lw=1.0,
            zorder=6
        )
        ax.text(
            xx,
            cy - 0.030 * scale,
            lab,
            ha="center",
            va="top",
            fontsize=6.5,
            color="#4E79A7",
            zorder=6
        )

    # phase blocks above axis
    colors = ["#C7E9C0", "#A1D99B", "#74C476"]

    for i in range(3):
        ax.add_patch(
            patches.Rectangle(
                (
                    cx - 0.050 * scale + i * 0.035 * scale,
                    cy + 0.025 * scale
                ),
                0.026 * scale,
                0.018 * scale,
                facecolor=colors[i],
                edgecolor="white",
                linewidth=0.5,
                zorder=6
            )
        )


def draw_table_icon(ax, cx, cy, scale=1.0):
    """
    Icon for A-B-C-D contingency table.
    """
    w = 0.100 * scale
    h = 0.075 * scale
    x = cx - w / 2
    y = cy - h / 2

    ax.add_patch(
        patches.Rectangle(
            (x, y),
            w,
            h,
            facecolor="white",
            edgecolor="#333333",
            linewidth=1.0,
            zorder=6
        )
    )

    ax.plot(
        [x + w / 2, x + w / 2],
        [y, y + h],
        color="#333333",
        lw=0.75,
        zorder=7
    )

    ax.plot(
        [x, x + w],
        [y + h / 2, y + h / 2],
        color="#333333",
        lw=0.75,
        zorder=7
    )

    labels = [
        ("A", x + w * 0.25, y + h * 0.75, "#D62728"),
        ("B", x + w * 0.75, y + h * 0.75, "#D62728"),
        ("C", x + w * 0.25, y + h * 0.25, "#4E79A7"),
        ("D", x + w * 0.75, y + h * 0.25, "#4E79A7"),
    ]

    for lab, tx, ty, col in labels:
        ax.text(
            tx,
            ty,
            lab,
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color=col,
            zorder=8
        )


def draw_heatmap_icon(ax, cx, cy, scale=1.0):
    """
    Icon for phase-lag heatmap and significance.
    """
    w = 0.110 * scale
    h = 0.075 * scale
    x = cx - w / 2
    y = cy - h / 2

    colors = [
        ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"],
        ["#67A9CF", "#F7F7F7", "#EF8A62", "#F7F7F7", "#B2182B"],
        ["#F7F7F7", "#67A9CF", "#F7F7F7", "#EF8A62", "#EF8A62"],
    ]

    nr = len(colors)
    nc = len(colors[0])

    for i in range(nr):
        for j in range(nc):
            ax.add_patch(
                patches.Rectangle(
                    (
                        x + j * w / nc,
                        y + (nr - 1 - i) * h / nr
                    ),
                    w / nc,
                    h / nr,
                    facecolor=colors[i][j],
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=6
                )
            )

    ax.add_patch(
        patches.Rectangle(
            (x, y),
            w,
            h,
            facecolor="none",
            edgecolor="#333333",
            linewidth=1.0,
            zorder=7
        )
    )

    ax.scatter(
        [x + w * 0.72, x + w * 0.90],
        [y + h * 0.68, y + h * 0.35],
        s=8 * scale,
        color="black",
        zorder=8
    )


# ============================================================
# Main drawing
# ============================================================

set_clean_style()

fig, ax = plt.subplots(figsize=(13.0, 5.4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Palette
blue = "#DCEAF7"
green = "#DFF2DF"
orange = "#FCE7C8"
purple = "#E8DFF1"

# Layout
card_y = 0.34
card_w = 0.205
card_h = 0.42
xs = [0.045, 0.285, 0.525, 0.765]

# Cards
add_card(
    ax,
    xs[0], card_y, card_w, card_h,
    1,
    "Input data",
    "Daily AR and EP fields\nENSO / MJO indices",
    blue
)

add_card(
    ax,
    xs[1], card_y, card_w, card_h,
    2,
    "Lag alignment",
    "ENSO: 0-6 month lags\nMJO: 0-30 day lags",
    green
)

add_card(
    ax,
    xs[2], card_y, card_w, card_h,
    3,
    "Phase metrics",
    "Aggregate A-D counts\nEstimate RD, RR, AF, PAF",
    orange
)

add_card(
    ax,
    xs[3], card_y, card_w, card_h,
    4,
    "Inference",
    "Phase contrasts\nBootstrap and FDR",
    purple
)

# ------------------------------------------------------------
# Icons: fixed bottom zone of each card
# ------------------------------------------------------------

icon_y = card_y + 0.085

draw_input_icon(
    ax,
    xs[0] + card_w / 2,
    icon_y,
    scale=1.05
)

draw_lag_icon(
    ax,
    xs[1] + card_w / 2,
    icon_y,
    scale=1.05
)

draw_table_icon(
    ax,
    xs[2] + card_w / 2,
    icon_y,
    scale=1.05
)

draw_heatmap_icon(
    ax,
    xs[3] + card_w / 2,
    icon_y,
    scale=1.05
)

# Arrows
for i in range(3):
    add_arrow(
        ax,
        xs[i] + card_w + 0.008,
        card_y + card_h / 2,
        xs[i + 1] - 0.008,
        card_y + card_h / 2
    )


# Subtitle
ax.text(
    0.5,
    0.865,
    "AR-EP association metrics are recomputed within each region, season, climate phase and lag.",
    ha="center",
    va="center",
    fontsize=9.5,
    color="#444444"
)

# Bottom concise statement
bottom_box = patches.FancyBboxPatch(
    (0.16, 0.115),
    0.68,
    0.085,
    boxstyle="round,pad=0.015,rounding_size=0.018",
    linewidth=0.9,
    edgecolor="#BDBDBD",
    facecolor="#F8F8F8"
)
ax.add_patch(bottom_box)

ax.text(
    0.5,
    0.157,
    "Core test:  Delta metric = metric(phase 1) - metric(phase 2)",
    ha="center",
    va="center",
    fontsize=9.5,
    fontweight="bold",
    color="#333333"
)

# Output note
ax.text(
    0.5,
    0.055,
    "Outputs: phase summaries, pairwise contrasts, rankings, ENSO lag curves and MJO phase-lag heatmaps",
    ha="center",
    va="center",
    fontsize=8.2,
    color="#555555"
)

plt.tight_layout()
plt.show()
