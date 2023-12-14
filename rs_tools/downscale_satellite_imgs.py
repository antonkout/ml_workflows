import rasterio
from skimage.transform import rescale

# Set the filepath for the input Sentinel-2 image
filepath = 'path/to/sentinel2_image.tiff'

# Open the image with rasterio
with rasterio.open(filepath) as src:
    # Read the image data into a NumPy array
    img = src.read()

    # Get the image metadata
    meta = src.meta

# Set the new resolution of the image (1m in this case)
new_resolution = (1, 1)

# Calculate the scaling factor for the image
scaling_factor = (new_resolution[0] / src.res[0], new_resolution[1] / src.res[1])

# Downscale the image using the rescale function from skimage
downscaled_img = rescale(img, scaling_factor, order=0, preserve_range=True, multichannel=True, anti_aliasing=False)

# Update the metadata for the downscaled image
meta.update(
    driver='GTiff',
    dtype=rasterio.uint16,
    count=img.shape[0],
    width=downscaled_img.shape[2],
    height=downscaled_img.shape[1],
    transform=src.transform * src.transform.scale(scaling_factor[0], scaling_factor[1]),
    crs=src.crs
)

# Set the filepath for the output downscaled image
output_filepath = 'path/to/downscaled_image.tiff'

# Save the downscaled image with rasterio
with rasterio.open(output_filepath, 'w', **meta) as dst:
    dst.write(downscaled_img)
