import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =========================
# 1. load data
# =========================
df = pd.read_csv("data/ABCD.csv")  # your path file

# =========================
# 2. give latitude and longitude if they are none
# =========================
lats = np.arange(10, 60.0001, 0.25)
lons = np.arange(100, 150.0001, 0.25)
lon_grid, lat_grid = np.meshgrid(lons, lats)

df["lat"] = lat_grid.ravel()
df["lon"] = lon_grid.ravel()

# =========================
# 3. calculate
# =========================
df["N"] = df["A"] + df["B"] + df["C"] + df["D"]

df["p_AR"] = (df["A"] + df["B"]) / df["N"]
df["p_notAR"] = (df["C"] + df["D"]) / df["N"]
df["p_E_AR"] = df["A"] / (df["A"] + df["B"]).replace(0, np.nan)
df["p_E_notAR"] = df["C"] / (df["C"] + df["D"]).replace(0, np.nan)

df["Freq_E"] = (df["A"] + df["C"]) / df["N"]
df["Contrib_AR"] = df["A"] / df["N"]
df["Contrib_notAR"] = df["C"] / df["N"]

# Attributable Fraction : AF
df["AF"] = (df["p_E_AR"] - df["p_E_notAR"]) / df["p_E_AR"]

# Population Attributable Fraction : PAF
df["PAF"] = (df["Freq_E"] - df["p_notAR"] * df["p_E_notAR"]) / df["Freq_E"]

# =========================
# 4. Significance test
# =========================
p_values = []
for _, row in df.iterrows():
    table = [[row["A"], row["B"]],
             [row["C"], row["D"]]]
    _, p = fisher_exact(table, alternative="greater")
    p_values.append(p)

df["p_value"] = p_values
df["signif"] = df["p_value"] < 0.05

# =========================
# 5. results
# =========================
nlat, nlon = len(lats), len(lons)
PAF_grid = df["PAF"].values.reshape(nlat, nlon)
AF_grid = df["AF"].values.reshape(nlat, nlon)
signif_grid = df["signif"].values.reshape(nlat, nlon)

# =========================
# 6. give significant grid
# =========================
PAF_sig = np.where(signif_grid, PAF_grid, np.nan)
AF_sig = np.where(signif_grid, AF_grid, np.nan)

# =========================
# 7. global average
# =========================
mean_PAF = np.nanmean(PAF_sig)
mean_AF = np.nanmean(AF_sig)

print(f"Global mean PAF (p<0.05): {mean_PAF:.4f}")
print(f"Global mean AF  (p<0.05): {mean_AF:.4f}")

# =========================
# 8. draw map
# =========================
def plot_map(data_grid, title, outfile, cmap="coolwarm"):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([100, 150, 10, 60], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)

    mesh = ax.pcolormesh(lons, lats, data_grid, cmap=cmap, shading="auto",
                         transform=ccrs.PlateCarree())
    cb = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
    cb.set_label(title)

    plt.title(title, fontsize=14)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# 9. output the map
# =========================
plot_map(PAF_sig, "Population Attributable Fraction (PAF, p<0.05)", "PAF_map.png")
plot_map(AF_sig, "Attributable Fraction (AF, p<0.05)", "AF_map.png")
