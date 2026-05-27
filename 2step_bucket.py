import matplotlib

matplotlib.use('TkAgg')
import netCDF4 as nc
import numpy as np
from netCDF4 import num2date, date2num
from datetime import datetime
from scipy.stats import fisher_exact, chi2
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import textwrap

grid_aggregate_size = 1
sig_level = 0.05
lift_cap_percentile = 95
lift_max_val = 10
output_dir = "G:/ar_analysis/output_1"
os.makedirs(output_dir, exist_ok=True)
control_modes = ["ENSO", "MJO", "AO", "PDO"]
do_heterogeneity_test = True

REGIONS = {
    "South_China": {'lon_min': 105, 'lon_max': 125, 'lat_min': 18, 'lat_max': 32},
    "North_China": {'lon_min': 105, 'lon_max': 125, 'lat_min': 32, 'lat_max': 42},
    "Northeast_Asia": {'lon_min': 120, 'lon_max': 140, 'lat_min': 40, 'lat_max': 58},
    "Japan_Korea": {'lon_min': 125, 'lon_max': 145, 'lat_min': 30, 'lat_max': 45}
}


def load_enso():
    enso_file = "G:/precipitation/nino3.4.csv"
    # Read data starting from the second row, with the first column as the index
    df = pd.read_csv(enso_file, header=1, index_col=0)
    # Convert index to datetime objects, handling mixed date formats
    df.index = pd.to_datetime(df.index, errors='coerce', format='mixed')
    df = df.dropna(subset=[df.columns[0]])
    # Get nino3.4 values
    values = df.iloc[:, 0].astype(float)
    # Categorize values into buckets
    bucket = pd.Series(index=values.index, dtype="object")
    bucket[values > 0.5] = "ElNino"
    bucket[values < -0.5] = "LaNina"
    bucket[(values >= -0.5) & (values <= 0.5)] = "Neutral"
    return bucket


def load_mjo():
    mjo_file = "G:/precipitation/MJO.csv"
    # Read data starting from the third row
    df = pd.read_csv(mjo_file, header=2)
    # Ensure correct column names
    df.columns = ['year', 'month', 'day', 'RMM1', 'RMM2', 'RMM_phase', 'RMM_amplitude', 'RMM_weight', "Column9",
                  "Column10"]
    # Combine year, month, and day columns to create a timestamp index
    df['time'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    df = df.dropna(subset=['time'])
    df = df.set_index('time')
    # Bucket categorization: inactive / active-1to4 / active-5to8
    amp = df['RMM_amplitude']
    phase = df['RMM_phase']
    bucket = pd.Series(index=df.index, dtype="object")
    bucket[amp < 1] = "Inactive"
    bucket[(amp >= 1) & (phase.isin([1, 2, 3, 4]))] = "Active_1to4"
    bucket[(amp >= 1) & (phase.isin([5, 6, 7, 8]))] = "Active_5to8"
    return bucket


def load_ao():
    ao_file = "G:/precipitation/AO.csv"
    df = pd.read_csv(ao_file)
    df['time'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')
    df = df.set_index('time')

    values = df['index'].astype(float)
    bucket = pd.Series(index=values.index, dtype="object")
    bucket[values > 0.5] = "Positive"
    bucket[values < -0.5] = "Negative"
    bucket[(values >= -0.5) & (values <= 0.5)] = "Neutral"
    return bucket


def load_pdo():
    pdo_file = "G:/precipitation/PDO.csv"
    df = pd.read_csv(pdo_file)

    df_long = pd.melt(df, id_vars=['Year'], var_name='Month', value_name='index')

    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df_long['MonthNum'] = df_long['Month'].map(month_map)

    df_long['time'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['MonthNum'].astype(str),
                                     format='%Y-%m')
    df_long = df_long.set_index('time').sort_index()

    values = df_long['index'].astype(float)
    bucket = pd.Series(index=values.index, dtype="object")
    bucket[values > 0.5] = "Positive"
    bucket[values < -0.5] = "Negative"
    bucket[(values >= -0.5) & (values <= 0.5)] = "Neutral"
    return bucket


def get_bucket_labels(dates_series):
    labels = {}
    if "ENSO" in control_modes:
        enso_bucket = load_enso()
        labels["ENSO"] = enso_bucket.reindex(dates_series, method='nearest').values
    if "MJO" in control_modes:
        mjo_bucket = load_mjo()
        labels["MJO"] = mjo_bucket.reindex(dates_series, method='nearest').values
    if "AO" in control_modes:
        ao_bucket = load_ao()
        labels["AO"] = ao_bucket.reindex(dates_series, method='nearest').values
    if "PDO" in control_modes:
        pdo_bucket = load_pdo()
        labels["PDO"] = pdo_bucket.reindex(dates_series, method='nearest').values
    return labels


def compute_stats(ar_mask, pr_mask):
    A = np.sum((ar_mask == 1) & (pr_mask == 1))
    B = np.sum((ar_mask == 1) & (pr_mask == 0))
    C = np.sum((ar_mask == 0) & (pr_mask == 1))
    D = np.sum((ar_mask == 0) & (pr_mask == 0))
    table = [[A, B], [C, D]]

    if (A + B) == 0 or (C + D) == 0 or A == 0 or B == 0 or C == 0 or D == 0:
        return {"OR": np.nan, "lift": np.nan, "pval": 1, "table": (A, B, C, D), "log_OR": np.nan, "SE": np.nan}

    OR = (A * D) / (B * C)
    p_ext_ar = A / (A + B)
    p_ext_no_ar = C / (C + D)
    lift = p_ext_ar / p_ext_no_ar if p_ext_no_ar > 0 else np.nan
    _, pval = fisher_exact(table)

    log_OR = np.log(OR)
    SE = np.sqrt(1 / A + 1 / B + 1 / C + 1 / D)

    return {"OR": OR, "lift": lift, "pval": pval, "table": (A, B, C, D), "log_OR": log_OR, "SE": SE}


def run_analysis(ar_data_subset, precip_data_subset, ar_dates_subset, bucket_labels):
    results = {}

    if not bucket_labels:
        results['overall'] = compute_stats(ar_data_subset, precip_data_subset)
        return results

    import itertools

    active_buckets = {k: np.unique(v) for k, v in bucket_labels.items() if v is not None}
    bucket_names = list(active_buckets.keys())
    bucket_values_combinations = list(itertools.product(*active_buckets.values()))

    for combo in tqdm(bucket_values_combinations, desc="    Processing combinations", leave=False):
        mask = np.full(ar_data_subset.shape[0], True)
        label_parts = []
        for i, bucket_name in enumerate(bucket_names):
            mask &= (bucket_labels[bucket_name] == combo[i])
            label_parts.append(f"{bucket_name}={combo[i]}")

        label = ", ".join(label_parts)
        if np.any(mask):
            results[label] = compute_stats(ar_data_subset[mask], precip_data_subset[mask])

    if do_heterogeneity_test and len(results) > 1:
        valid_results = [r for r in results.values() if
                         "OR" in r and not np.isnan(r["OR"]) and 'table' in r and "SE" in r and not np.isnan(r["SE"])]
        if len(valid_results) > 1:
            weights = [1 / (r["SE"] ** 2) for r in valid_results]
            log_ORs = [r["log_OR"] for r in valid_results]
            avg_log_OR = np.average(log_ORs, weights=weights)
            Q = np.sum(weights * ((np.array(log_ORs) - avg_log_OR) ** 2))
            df = len(valid_results) - 1
            if Q > df:
                pQ = 1 - chi2.cdf(Q, df)
                I2 = max(0, (Q - df) / Q) * 100
            else:
                pQ = 1
                I2 = 0
            results["heterogeneity"] = {"Q": Q, "pQ": pQ, "I2": I2}

    return results


def run_logistic_regression(ar_data, precip_data, dates, enso_bucket, mjo_bucket, ao_bucket, pdo_bucket):
    print("\nRunning Logistic Regression...")
    print("  ar_data shape:", ar_data.shape)
    print("  precip_data shape:", precip_data.shape)
    if ar_data.shape != precip_data.shape:
        raise ValueError(f"Shape mismatch: ar_data.shape={ar_data.shape}, precip_data.shape={precip_data.shape}")
    from datetime import datetime
    dates_dt = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dates]
    if len(dates_dt) != ar_data.shape[0]:
        raise ValueError(f"Dates length ({len(dates_dt)}) does not match ar_data time dimension ({ar_data.shape[0]})")
    ar_flat = ar_data.flatten()
    precip_flat = precip_data.flatten()
    total_size = ar_flat.size
    expected_size = np.prod(ar_data.shape)
    if total_size != expected_size:
        raise ValueError(f"total_size ({total_size}) does not match np.prod(ar_data.shape) ({expected_size})")
    print(f"  Total size of flattened array: {total_size}")
    print("  Preparing data for regression model...")
    sample_fraction = 0.1
    max_sample_size = 5000000
    sample_size = min(int(total_size * sample_fraction), max_sample_size)
    print(f"  Sample size: {sample_size}")
    if total_size <= sample_size:
        sample_indices = np.arange(total_size, dtype=np.int64)
    else:
        sample_set = set()
        factor = 2
        while len(sample_set) < sample_size:
            needed = sample_size - len(sample_set)
            new_indices = np.random.randint(0, total_size, size=int(needed * factor), dtype=np.int64)
            sample_set.update(map(int, new_indices))
        sample_indices = np.fromiter(sample_set, dtype=np.int64)
        if sample_indices.size > sample_size:
            sel = np.random.choice(sample_indices.size, size=sample_size, replace=False)
            sample_indices = sample_indices[sel]
    if np.any(sample_indices < 0) or np.any(sample_indices >= total_size):
        raise ValueError(f"Invalid sample_indices: min={np.min(sample_indices)}, max={np.max(sample_indices)}, total_size={total_size}")
    print(f"  sample_indices min: {np.min(sample_indices)}, max: {np.max(sample_indices)}")
    import pandas as pd
    df_data = pd.DataFrame({'EP': precip_flat[sample_indices], 'AR': ar_flat[sample_indices]})
    try:
        time_indices, lat_indices, lon_indices = np.unravel_index(sample_indices, ar_data.shape)
    except ValueError as e:
        print(f"Error in np.unravel_index: {e}")
        print(f"ar_data.shape: {ar_data.shape}, sample_indices shape: {sample_indices.shape}")
        raise

    df_data['date'] = [dates_dt[i] for i in time_indices]
    if "ENSO" in control_modes:
        df_data['ENSO'] = enso_bucket.reindex(df_data['date'], method='nearest').values
    if "MJO" in control_modes:
        df_data['MJO'] = mjo_bucket.reindex(df_data['date'], method='nearest').values
    if "AO" in control_modes:
        df_data['AO'] = ao_bucket.reindex(df_data['date'], method='nearest').values
    if "PDO" in control_modes:
        df_data['PDO'] = pdo_bucket.reindex(df_data['date'], method='nearest').values

    df_data['month'] = df_data['date'].dt.month
    df_data['trend'] = df_data.index
    dummy_cols = [mode for mode in control_modes if mode in df_data.columns]
    df_data = pd.get_dummies(df_data, columns=dummy_cols, drop_first=True, dtype=int)
    exog_vars = ['AR', 'trend']
    for mode in control_modes:
        for col in [c for c in df_data.columns if c.startswith(f'{mode}_')]:
            interaction_term = f'AR_{col}'
            df_data[interaction_term] = df_data['AR'] * df_data[col]
            exog_vars.extend([col, interaction_term])

    df_data = pd.get_dummies(df_data, columns=['month'], drop_first=True, dtype=int)
    exog_vars.extend([col for col in df_data.columns if col.startswith('month_')])

    exog_vars = sorted(list(set(exog_vars)))
    import statsmodels.api as sm
    df_data = sm.add_constant(df_data, has_constant='add')
    exog_vars.insert(0, 'const')

    print("  Fitting regression model...")
    model = sm.Logit(df_data['EP'], df_data[exog_vars])
    result = model.fit(disp=False)
    print(result.summary())

    print("\nWald test p-values for interaction terms:")
    for var in result.pvalues.index:
        if var.startswith('AR_'):
            print(f"  {var}: {result.pvalues[var]:.4f}")
    import matplotlib.pyplot as plt
    import os
    num_plots = len(control_modes)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), squeeze=False)
    axes = axes.flatten()
    plot_idx = 0

    def plot_or(ax, mode_name, base_case, params):
        or_dict = {base_case: np.exp(params['AR'])}
        for col, term in [(k.split('_')[-1], k) for k in params.index if k.startswith(f'AR_{mode_name}_')]:
            or_dict[col] = np.exp(params['AR'] + params[term])

        sorted_keys = sorted(or_dict.keys())
        ax.bar(sorted_keys, [or_dict[k] for k in sorted_keys])
        ax.set_title(f'AR-Precipitation OR by {mode_name} Phase')
        ax.set_ylabel('Odds Ratio')
        ax.axhline(1, color='red', linestyle='--')

    if "ENSO" in control_modes:
        plot_or(axes[plot_idx], 'ENSO', 'ElNino', result.params)
        plot_idx += 1
    if "MJO" in control_modes:
        plot_or(axes[plot_idx], 'MJO', 'Inactive', result.params)
        plot_idx += 1
    if "AO" in control_modes:
        plot_or(axes[plot_idx], 'AO', 'Negative', result.params)
        plot_idx += 1
    if "PDO" in control_modes:
        plot_or(axes[plot_idx], 'PDO', 'Negative', result.params)
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "marginal_effects.png"))
    print(f"\nMarginal effects plot saved to: {os.path.join(output_dir, 'marginal_effects.png')}")
    plt.show()


def run_regional_analysis(ar_data, precip_data, ar_dates_sel, ar_lats, ar_lons):
    print("\nRunning regional analysis...")

    ar_dates_sel_dt = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in ar_dates_sel]
    bucket_labels = get_bucket_labels(pd.to_datetime(ar_dates_sel_dt))

    # Store results for all regions
    all_results = {}

    for region_name, region_coords in tqdm(REGIONS.items(), desc="Analyzing Regions"):
        lat_mask = (ar_lats >= region_coords['lat_min']) & (ar_lats <= region_coords['lat_max'])
        lon_mask = (ar_lons >= region_coords['lon_min']) & (ar_lons <= region_coords['lon_max'])

        if not np.any(lat_mask) or not np.any(lon_mask):
            print(f"Skipping {region_name}: No grid points found.")
            continue

        region_ar_data = ar_data[:, lat_mask, :][:, :, lon_mask]
        region_precip_data = precip_data[:, lat_mask, :][:, :, lon_mask]

        if region_ar_data.size == 0 or region_precip_data.size == 0:
            print(f"Skipping {region_name}: Data is empty.")
            continue

        region_results = run_analysis(region_ar_data, region_precip_data, ar_dates_sel, bucket_labels)
        all_results[region_name] = region_results

        # Plot forest plot for each region individually (optional)
        plot_forest_plot(region_results, title=f"Forest Plot for {region_name}",
                         filename=f"forest_plot_{region_name}.png")

    if all_results:
        plot_combined_forest_plot(all_results, title="Combined Forest Plot for All Regions",
                                 filename="combined_forest_plot.png")
        plot_grouped_scatter(all_results, title="Grouped Scatter Plot for All Regions",
                             filename="grouped_scatter_plot.png")
    else:
        print("No regional data available to plot combined figures.")


def plot_forest_plot(results, title="Forest Plot of Odds Ratios", filename="forest_plot.png"):
    subgroup_results = {k: v for k, v in results.items() if k not in ["overall", "heterogeneity"]}

    if not subgroup_results:
        return

    plot_data = {k: v for k, v in subgroup_results.items() if
                 "OR" in v and not np.isnan(v["OR"]) and "SE" in v and not np.isnan(v["SE"])}

    if not plot_data:
        return

    labels = list(plot_data.keys())
    log_ORs = np.array([v["log_OR"] for v in plot_data.values()])
    SEs = np.array([v["SE"] for v in plot_data.values()])

    lower_bounds = np.exp(log_ORs - 1.96 * SEs)
    upper_bounds = np.exp(log_ORs + 1.96 * SEs)
    ORs = np.exp(log_ORs)

    sort_idx = np.argsort(ORs)[::-1]
    ORs, labels, lower_bounds, upper_bounds = ORs[sort_idx], [labels[i] for i in sort_idx], lower_bounds[sort_idx], \
    upper_bounds[sort_idx]

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
        text_str = (f"Heterogeneity Test:\n"
                    f"Cochran's Q = {het_stats['Q']:.2f}\n"
                    f"p-value = {het_stats['pQ']:.3f}\n"
                    f"I² = {het_stats['I2']:.1f}%")
        ax.text(1.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def plot_combined_forest_plot(all_results, title="Combined Forest Plot of Odds Ratios",
                              filename="combined_forest_plot.png"):
    """
    Plots an optimized combined forest plot, sharing the left-side bucket labels,
    with independent OR and confidence intervals for each region.

    Args:
        all_results (dict): A dictionary containing results for each region, in the format {region_name: results_dict}
        title (str): The plot title
        filename (str): The output filename
    """
    # Collect all bucket labels
    all_labels = set()
    for region_name, results in all_results.items():
        subgroup_results = {k: v for k, v in results.items() if k not in ["overall", "heterogeneity"]}
        all_labels.update(subgroup_results.keys())

    labels = sorted(list(all_labels))  # Unified sorted bucket labels
    if not labels:
        print("No valid bucket labels, cannot plot combined forest plot.")
        return

    # Dynamically adjust plot size
    num_regions = len(all_results)
    num_labels = len(labels)
    fig_height = max(8, num_labels * 0.5)  # Adjust height based on number of labels
    fig_width = max(12, 4 * num_regions + 4)  # Increase width to accommodate labels
    fig, axes = plt.subplots(1, num_regions, figsize=(fig_width, fig_height), sharey=True)

    # If there is only one region, axes is not an array, so convert it to a list
    if num_regions == 1:
        axes = [axes]

    for idx, (region_name, results) in enumerate(tqdm(all_results.items(), desc="Plotting regions")):
        ax = axes[idx]
        subgroup_results = {k: v for k, v in results.items() if k not in ["overall", "heterogeneity"]}
        plot_data = {k: v for k, v in subgroup_results.items() if
                     "OR" in v and not np.isnan(v["OR"]) and "SE" in v and not np.isnan(v["SE"])}

        # Initialize ORs and confidence intervals
        log_ORs = np.full(len(labels), np.nan)
        SEs = np.full(len(labels), np.nan)
        ORs = np.full(len(labels), np.nan)
        lower_bounds = np.full(len(labels), np.nan)
        upper_bounds = np.full(len(labels), np.nan)

        # Fill in valid data
        for i, label in enumerate(labels):
            if label in plot_data:
                log_ORs[i] = plot_data[label]["log_OR"]
                SEs[i] = plot_data[label]["SE"]
                ORs[i] = np.exp(log_ORs[i])
                lower_bounds[i] = np.exp(log_ORs[i] - 1.96 * SEs[i])
                upper_bounds[i] = np.exp(log_ORs[i] + 1.96 * SEs[i])

        # Plot only non-NaN data
        valid_idx = ~np.isnan(ORs)
        if not np.any(valid_idx):
            ax.text(0.5, 0.5, f"No valid data for {region_name}", ha='center', va='center', transform=ax.transAxes)
            continue

        y_pos = np.arange(len(labels))
        ax.scatter(ORs[valid_idx], y_pos[valid_idx], color='blue', zorder=2, s=50)
        ax.hlines(y_pos[valid_idx], lower_bounds[valid_idx], upper_bounds[valid_idx], color='gray', linestyle='-',
                  zorder=1)
        ax.axvline(1, color='black', linestyle='--', linewidth=1)

        # Set title and labels
        ax.set_title(region_name, fontsize=12)
        ax.set_xlabel("Odds Ratio (OR)", fontsize=10)
        ax.set_xscale('log')
        ax.grid(True, which="both", axis="x", linestyle='--', linewidth=0.5)

        # Set Y-axis labels only on the first subplot
        if idx == 0:
            # Wrap long labels automatically
            wrapped_labels = [textwrap.fill(label, width=30) for label in labels]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(wrapped_labels, fontsize=8)
        else:
            ax.set_yticks(y_pos)
            ax.set_yticklabels([])

        # Add heterogeneity statistics
        if "heterogeneity" in results:
            het_stats = results["heterogeneity"]
            text_str = (f"Heterogeneity:\n"
                        f"Q = {het_stats['Q']:.2f}\n"
                        f"p = {het_stats['pQ']:.3f}\n"
                        f"I² = {het_stats['I2']:.1f}%")
            ax.text(1.05, 0.95, text_str, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

    # Set overall title and layout
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Increase resolution and ensure labels are not cropped
    print(f"Optimized combined forest plot saved to: {output_path}")
    plt.close(fig)


def plot_grouped_scatter(all_results, title="Grouped Scatter Plot of Odds Ratios", filename="grouped_scatter_plot.png"):
    """
    Plots a grouped scatter plot, with different colors/markers for each region,
    sharing the bucket labels and OR axis.

    Args:
        all_results (dict): A dictionary containing results for each region, in the format {region_name: results_dict}
        title (str): The plot title
        filename (str): The output filename
    """
    # Collect all bucket labels
    all_labels = set()
    for region_name, results in all_results.items():
        subgroup_results = {k: v for k, v in results.items() if k not in ["overall", "heterogeneity"]}
        all_labels.update(subgroup_results.keys())

    labels = sorted(list(all_labels))
    if not labels:
        print("No valid bucket labels, cannot plot grouped scatter plot.")
        return

    # Set colors and markers
    colors = plt.cm.get_cmap('tab10', len(all_results))
    markers = ['o', 's', '^', 'D']  # Different marker styles
    num_regions = len(all_results)
    fig_height = max(8, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos_base = np.arange(len(labels))
    offset = np.linspace(-0.3, 0.3, num_regions)  # Set an offset for each region

    for idx, (region_name, results) in enumerate(tqdm(all_results.items(), desc="Plotting regions")):
        subgroup_results = {k: v for k, v in results.items() if k not in ["overall", "heterogeneity"]}
        plot_data = {k: v for k, v in subgroup_results.items() if
                     "OR" in v and not np.isnan(v["OR"]) and "SE" in v and not np.isnan(v["SE"])}

        ORs = np.full(len(labels), np.nan)
        lower_bounds = np.full(len(labels), np.nan)
        upper_bounds = np.full(len(labels), np.nan)

        for i, label in enumerate(labels):
            if label in plot_data:
                log_OR = plot_data[label]["log_OR"]
                SE = plot_data[label]["SE"]
                ORs[i] = np.exp(log_OR)
                lower_bounds[i] = np.exp(log_OR - 1.96 * SE)
                upper_bounds[i] = np.exp(log_OR + 1.96 * SE)

        valid_idx = ~np.isnan(ORs)
        y_pos = y_pos_base + offset[idx]
        ax.scatter(ORs[valid_idx], y_pos[valid_idx], color=colors(idx), marker=markers[idx % len(markers)],
                   label=region_name, s=50, zorder=2)
        for i in np.where(valid_idx)[0]:
            ax.plot([lower_bounds[i], upper_bounds[i]], [y_pos[i], y_pos[i]], color=colors(idx), linestyle='-',
                    zorder=1)

    ax.axvline(1, color='black', linestyle='--', linewidth=1)
    wrapped_labels = [textwrap.fill(label, width=30) for label in labels]
    ax.set_yticks(y_pos_base)
    ax.set_yticklabels(wrapped_labels, fontsize=8)
    ax.set_xlabel("Odds Ratio (OR)", fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, which="both", axis="x", linestyle='--', linewidth=0.5)
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grouped scatter plot saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading control bucket data...")
    enso_bucket = load_enso() if "ENSO" in control_modes else None
    mjo_bucket = load_mjo() if "MJO" in control_modes else None
    ao_bucket = load_ao() if "AO" in control_modes else None
    pdo_bucket = load_pdo() if "PDO" in control_modes else None

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

    start_date_input = input("Start Date (YYYY-MM-DD HH:M:SS): ")
    end_date_input = input("End Date (YYYY-MM-DD HH:M:SS): ")
    start_date = datetime.strptime(start_date_input, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(end_date_input, "%Y-%m-%d %H:%M:%S")

    print("Aligning time dimension...")
    ar_dates_pd = pd.to_datetime(
        [d.strftime('%Y-%m-%d %H:%M:%S') for d in tqdm(ar_dates, desc="  Converting AR dates")])
    precip_dates_pd = pd.to_datetime(
        [d.strftime('%Y-%m-%d %H:%M:%S') for d in tqdm(precip_dates, desc="  Converting Precip dates")])

    df_ar = pd.DataFrame({'ar_idx': np.arange(len(ar_dates_pd))}, index=ar_dates_pd)
    df_precip = pd.DataFrame({'precip_idx': np.arange(len(precip_dates_pd))}, index=precip_dates_pd)

    df_ar_sel = df_ar[start_date:end_date]
    df_precip_sel = df_precip[start_date:end_date]

    merged_df = df_ar_sel.merge(df_precip_sel, left_index=True, right_index=True, how='inner')

    if merged_df.empty:
        print("Error: No common timestamps found between the two data files within the specified date range.")
        exit()

    ar_indices = merged_df['ar_idx'].values
    precip_indices = merged_df['precip_idx'].values

    ar_dates_sel = ar_dates[ar_indices]

    lon_min, lon_max = 100, 150
    lat_min, lat_max = 10, 60


    def get_spatial_indices(ds, lat_var, lon_var):
        lats_all = ds.variables[lat_var][:]
        lons_all = ds.variables[lon_var][:]
        lat_mask = (lats_all >= lat_min) & (lats_all <= lat_max)
        lon_mask = (lons_all >= lon_min) & (lons_all <= lon_max)
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            return None, None, None, None
        return lats_all[lat_idx], lons_all[lon_idx], lat_idx, lon_idx


    print("Masking spatial dimension...")
    precip_lats, precip_lons, lat_indices, lon_indices = get_spatial_indices(precip_ds, 'latitude', 'longitude')
    if lat_indices is None:
        print(f"Error: No precipitation data grid points found within the specified lat/lon range (Lat: {lat_min}-{lat_max}, Lon: {lon_min}-{lon_max}).")
        exit()

    ar_lats, ar_lons, ar_lat_indices, ar_lon_indices = get_spatial_indices(ar_ds, 'lat', 'lon')
    if ar_lat_indices is None:
        print(f"Error: No AR data grid points found within the specified lat/lon range (Lat: {lat_min}-{lat_max}, Lon: {lon_min}-{lon_max}).")
        exit()

    if ar_lats.size > 1 and ar_lats[0] > ar_lats[-1]: ar_lats, ar_lat_indices = ar_lats[::-1], ar_lat_indices[::-1]
    if precip_lats.size > 1 and precip_lats[0] > precip_lats[-1]: precip_lats, lat_indices = precip_lats[
                                                                                             ::-1], lat_indices[::-1]

    print("Loading primary data into memory...")
    ar_data = ar_ds.variables['ar_happen'][ar_indices, :, :][:, ar_lat_indices, :][:, :, ar_lon_indices].astype(np.int8)
    if 'extreme_precipitation_flag_ocean' in precip_ds.variables:
        precip_ocean = precip_ds.variables['extreme_precipitation_flag_ocean'][precip_indices, :, :][:, lat_indices, :][
                       :, :, lon_indices]
        precip_land = precip_ds.variables['extreme_precipitation_flag_land'][precip_indices, :, :][:, lat_indices, :][:,
                      :, lon_indices]
        precip_data = np.maximum(precip_ocean, precip_land).astype(np.int8)
    else:
        precip_data = precip_ds.variables['extreme_precipitation_flag'][precip_indices, :, :][:, lat_indices, :][:, :,
                      lon_indices].astype(np.int8)

    if any(mode in control_modes for mode in ["ENSO", "MJO", "AO", "PDO"]):
        run_logistic_regression(ar_data, precip_data, ar_dates_sel, enso_bucket, mjo_bucket, ao_bucket, pdo_bucket)

    run_regional_analysis(ar_data, precip_data, ar_dates_sel, ar_lats, ar_lons)

    ar_ds.close()
    precip_ds.close()
    print("\nAll computations complete, results have been saved.")
