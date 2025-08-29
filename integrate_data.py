import os
import xarray as xr
import numpy as np
import pandas as pd

base_path = r'G:\ERA5_grib\precipitation'
output_path = r'G:\precipitation\merged_precipitation.nc'

file_list = [
    os.path.join(base_path, '1983-1987.grib'),
    os.path.join(base_path, '1988-1992.grib'),
    os.path.join(base_path, '1993-1997.grib'),
    os.path.join(base_path, '1998-2002.grib'),
    os.path.join(base_path, '2003-2007.grib'),
    os.path.join(base_path, '2008-2012.grib'),
    os.path.join(base_path, '2013-2017.grib'),
    os.path.join(base_path, '2018-2022.grib'),
    os.path.join(base_path, '2023.grib'),
    os.path.join(base_path, '2024.grib'),
]

try:
    print("开始读取和拼接GRIB文件...")

    datasets = []
    for i, file_path in enumerate(file_list):
        print(f"正在处理文件: {file_path}")
        ds = xr.open_dataset(file_path, engine='cfgrib', chunks='auto')

        if i > 0:
            print("  - 删除第一个时间点...")
            ds = ds.isel(time=slice(1, None))

        datasets.append(ds)

    ds_combined = xr.concat(datasets, dim='time')

    time_sec = pd.to_datetime(ds_combined['time'].values).astype('datetime64[s]')
    _, index = np.unique(time_sec, return_index=True)
    ds_combined = ds_combined.isel(time=index)
    ds_combined = ds_combined.assign_coords(time=time_sec[index])
    ds_combined = ds_combined.sortby('time')

    print("\n时间范围：", ds_combined.time.min().values, "到", ds_combined.time.max().values)

    seconds_since_epoch = (
        (ds_combined.time.values - np.datetime64('1970-01-01T00:00:00')) /
        np.timedelta64(1, 's')
    ).astype('int64')
    ds_combined = ds_combined.assign_coords(time=("time", seconds_since_epoch))

    print("\n成功合并数据集，其信息如下：")
    print(ds_combined)
    attrs_to_remove = ['history', 'GRIB_edition', 'GRIB_centre',
                       'GRIB_centreDescription', 'GRIB_subCentre']
    for attr in attrs_to_remove:
        if attr in ds_combined.attrs:
            del ds_combined.attrs[attr]

    new_history = f"Created on {seconds_since_epoch.min()} to {seconds_since_epoch.max()} seconds since 1970-01-01 by combining multiple GRIB files."
    ds_combined.attrs['history'] = new_history
    ds_combined.attrs['Conventions'] = 'CF-1.7'
    ds_combined.attrs['institution'] = 'European Centre for Medium-Range Weather Forecasts'

    print("\n成功清理全局属性，并准备保存文件...")

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")

    encoding = {var: {"zlib": True, "complevel": 4} for var in ds_combined.data_vars}
    encoding["time"] = {"dtype": "int64"}

    ds_combined.to_netcdf(output_path, encoding=encoding)
    print("数据保存成功！")

except FileNotFoundError:
    print("错误：文件路径不正确，请检查输入GRIB文件是否存在。")
except Exception as e:
    print(f"发生了一个错误：{e}")
