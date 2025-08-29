import os
import xarray as xr
import pandas as pd
import numpy as np
import gc
from tqdm import tqdm


def calculate_extreme_threshold(nc_files, window_days=15, chunks={"time": 100}):
    ds = xr.open_mfdataset(nc_files, combine="by_coords", chunks=chunks)
    precip = ds["tp"]

    times = pd.to_datetime(precip.time.values)
    doy = times.dayofyear
    half_window = window_days // 2

    thresholds = []
    for day in range(1, 367):
        window_doys = [(day + i - 1) % 366 + 1 for i in range(-half_window, half_window + 1)]
        mask = np.isin(doy, window_doys)
        sel = precip.sel(time=mask)
        if sel.time.size > 0:
            thresh = sel.quantile(0.95, dim="time", skipna=True).compute()
            thresholds.append(thresh.expand_dims(dayofyear=[day]))

    ds.close()
    return xr.concat(thresholds, dim="dayofyear")


def detect_extremes_for_file(file_path, threshold_ds, output_dir):
    ds = xr.open_dataset(file_path, chunks={"time": 100})
    _, index = np.unique(ds.time, return_index=True)
    ds = ds.isel(time=index)

    times = pd.to_datetime(ds.time.values)
    doy = xr.DataArray(times.dayofyear, coords=[ds.time], dims="time")

    daily_thresh = threshold_ds.sel(dayofyear=doy)

    precip = ds["tp"]
    extreme_flag = (precip > daily_thresh["tp"])
    extreme_amount = precip.where(extreme_flag, other=np.nan)

    out_file = os.path.join(output_dir, "extreme_" + os.path.basename(file_path))
    xr.Dataset(
        {
            "extreme_precipitation_flag": (["time", "latitude", "longitude"], extreme_flag.astype(np.int8).values),
            "extreme_precipitation_amount": (["time", "latitude", "longitude"], extreme_amount.values)
        },
        coords={
            "time": precip["time"].values,
            "latitude": precip["latitude"].values,
            "longitude": precip["longitude"].values
        }
    ).to_netcdf(out_file)

    ds.close()
    del ds, extreme_flag, extreme_amount
    gc.collect()

    return out_file


def save_extreme_precipitation(final_ds, threshold_ds, output_path):
    extreme_flag = final_ds["extreme_precipitation_flag"]
    extreme_amount = final_ds["extreme_precipitation_amount"]

    extreme_count_grid = extreme_flag.sum(dim="time").compute()

    start_time = str(final_ds.time.values[0])
    end_time = str(final_ds.time.values[-1])

    ds_out = xr.Dataset(
        {
            "extreme_precipitation_flag": (["time", "latitude", "longitude"], extreme_flag.astype(np.int8).values),
            "extreme_precipitation_amount": (["time", "latitude", "longitude"], extreme_amount.values),
            "extreme_precipitation_count": (["latitude", "longitude"], extreme_count_grid.values),
            "threshold_90th": (["day_of_year", "latitude", "longitude"], threshold_ds["tp"].values)
        },
        coords={
            "time": final_ds["time"].values,
            "latitude": final_ds["latitude"].values,
            "longitude": final_ds["longitude"].values,
            "day_of_year": threshold_ds.dayofyear.values
        },
        attrs={
            "title": "Daily grid-wise extreme precipitation dataset",
            "summary": "Flag, count, amount, and daily threshold of extreme precipitation events",
            "time_coverage_start": start_time,
            "time_coverage_end": end_time,
            "institution": "Your Institution Name",
            "source": "ERA5 total precipitation data",
            "history": f"Created {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Conventions": "CF-1.6"
        }
    )
    ds_out.to_netcdf(output_path)
    print(f"✅ 已保存结果到: {output_path}")


def main(temp_dir, output_file):
    nc_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".nc")])

    threshold_file = os.path.join(temp_dir, "thresholds.nc")
    if not os.path.exists(threshold_file):
        threshold_ds = calculate_extreme_threshold(nc_files)
        threshold_ds.to_netcdf(threshold_file)
        del threshold_ds
        gc.collect()
        print(f"✅ 阈值文件已保存: {threshold_file}")
    else:
        print(f"📂 使用已有阈值文件: {threshold_file}")
    threshold_ds = xr.open_dataset(threshold_file, chunks={})

    print("⚙️ 正在逐月识别极端降水...")
    output_dir = os.path.join(temp_dir, "extremes")
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    for f in tqdm(nc_files, desc="处理文件", unit="file"):
        out_file = detect_extremes_for_file(f, threshold_ds, output_dir)
        saved_files.append(out_file)

    print("⚙️ 正在合并逐月结果...")
    final_ds = xr.open_mfdataset(saved_files, combine="by_coords", chunks={"time": 100})

    save_extreme_precipitation(final_ds, threshold_ds, output_file)

    final_ds.close()
    threshold_ds.close()
    del final_ds, threshold_ds
    gc.collect()

    print(f"🎉 最终结果已保存: {output_file}")


if __name__ == "__main__":
    temp_dir = r"G:\ar_analysis\temp\temp_precip_combined"
    output_file = r"G:\ar_analysis\data\extreme_precipitation.nc"
    main(temp_dir, output_file)
