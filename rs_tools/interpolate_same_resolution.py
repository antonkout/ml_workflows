import os
import rasterio
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject
from rasterio.mask import mask
import datetime as dt
from pathlib import Path

def resample_jp2s(input_folder, outpath, des_files, target_resolution, clipping, geometry):
    """
    Resamples all JP2 files in the input folder to the target resolution and saves them to the output folder.
    
    Parameters:
        input_folder (str): Path to the input folder.
        outpath (str): Path to the output folder.
        target_resolution (float): Target resolution in units of the input raster's CRS.
        clipping (bool, optional): Whether or not to clip the output rasters using a vector file. Defaults to False.
        vector (str, optional): Path to the vector file used for clipping. Required if clipping is True.
    """

    allfiles = os.listdir(input_folder)
    filenames = [file for file in allfiles for desired in des_files if desired in file]
    filenames.sort()

    # Specify the output
    dtime = filenames[0].split('_',-1)[1].split('T',-1)[0]
    outname = dt.datetime.strptime(dtime, '%Y%m%d').strftime('%Y-%m-%d')
    output = os.path.join(outpath, outname)
    os.makedirs(output, exist_ok=True)

    # Loop through all tifs in the input folder
    for file in filenames:
        if file.endswith('.jp2'):
            # Open the raster file
            with rasterio.open(os.path.join(input_folder, file)) as src:
                # Get the source crs, transform, and dimensions
                src_crs = src.crs
                src_transform = src.transform
                src_width = src.width
                src_height = src.height
                src_nodata = src.nodata

                # Calculate the transform and dimensions for the target resolution
                dst_transform, dst_width, dst_height = calculate_default_transform(
                                                        src_crs, src_crs,
                                                        src_width, src_height,
                                                        *src.bounds,
                                                        resolution=target_resolution)

                # Create the output file
                output_file = os.path.join(output, file)
                profile = src.profile
                profile.update({
                    'crs': src_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height
                })

                if clipping:
                    # from rasterio.io import MemoryFile
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**profile) as dst:
                            for i in range(1, src.count + 1):
                                reproject(
                                    source=rasterio.band(src, i),
                                    destination=rasterio.band(dst, i),
                                    src_transform=src_transform,
                                    src_crs=src_crs,
                                    dst_transform=dst_transform,
                                    dst_crs=src_crs,
                                    resampling=rasterio.enums.Resampling.nearest)
                            # Clip the output file using the vector file
                            out_image, out_transform = mask(dst, geometry, crop=True)
                            out_meta = dst.meta
                            out_meta.update({"driver": "GTiff",
                                            "height": out_image.shape[1],
                                            "width": out_image.shape[2],
                                            "transform": out_transform,
                                            })

                            # Write the clipped file to disk
                            with rasterio.open(output_file, "w", **out_meta) as dest:
                                dest.write(out_image)
                else:
                    # Resample the raster and write to the output file
                    with rasterio.open(output_file, 'w', **profile) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src_transform,
                                src_crs=src_crs,
                                dst_transform=dst_transform,
                                dst_crs=src_crs,
                                resampling=rasterio.enums.Resampling.nearest)

# Set the target resolution to 10 meters
target_resolution = (10, 10)

# Specify the input and output folders
folderpath = "/efs/prototyping/antonios/laos_deforestation/imagery/raw/2021-09-27/"
datefolders = [str(folder) for folder in Path(folderpath).glob("*") if folder.is_dir()]
des_files = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

# Set outpath directory
outpath = "/efs/prototyping/antonios/laos_deforestation/imagery/interpolated"

# Set clipping to True if you want to clip the resampled rasters into a specific AOI shapefile
clipping = True

# File to clip
vectorfile = "/efs/prototyping/antonios/laos_deforestation/aoi/laos_rubber_deforestation_aoi_32648_new.geojson"
vector = gpd.read_file(vectorfile)
geometry = vector.geometry.iloc[0]

for input_folder in datefolders:
    resample_jp2s(input_folder, outpath, des_files, target_resolution, clipping, geometry)