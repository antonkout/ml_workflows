"""
Script clusters any given satellite image by utilizing k-means
and calculates the zonal statistics of the generated classes.

created: 2022/12/09
author: Antonios Koutroumpas
email: antonkout@gmail.com
"""
import numpy as np
import rasterio
from loguru import logger
from timeit import default_timer as timer
from numba import jit, njit
import os

def gabor_process_threaded(img, freq, threadn):
    """The Gabor filter is a sinusoidal plane wave modulated by a Gaussian function, which is used to extract texture 
    features from images.
    ----------
    Parameters
    ----------
        img: np.array . Multi-dimensional numpy array in format (height, width, n_channels)
        freq: float . Frequency of a Gabor filter refers to the spatial frequency of the sinusoidal waveform that is used.
        threadn: int . Number of threads
    """
    from multiprocessing.pool import ThreadPool
    from threading import Lock
    time_script_start = timer()
    @jit
    def gabor_numpy(freq):
        # Define the size of the kernel
        size = (2, 2)
        # Define the standard deviation of the Gaussian envelope
        sigma = 1.0
        # Define the phase offset of the sinusoidal waveform
        phase = 0.0
        # Create grids of x and y values
        x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
        # Calculate the Gabor kernel
        gabor_kernel = np.exp(-((x ** 2) + (y ** 2)) / (2 * sigma ** 2)) * np.cos(2 * np.pi * freq * x + phase)
        return gabor_kernel

    original_shape = img.shape
    kernel = np.real(gabor_numpy(freq))
    accum = np.zeros_like(img).reshape(-1).astype(np.float64)
    accum_lock = Lock()
    imgn = img.reshape(-1).astype(np.float64)

    def f(kernel):
        fimg = np.convolve(imgn, kernel, mode='same')
        with accum_lock:
            np.maximum(accum, fimg, accum)
    pool = ThreadPool(processes=threadn)
    pool.map(f, kernel)
    time_script_end = timer()
    logger.debug(
        "---Gabor filtering execution time:\t%2.2f minutes"% np.round((time_script_end - time_script_start) / 60, 2),2)
    return np.reshape(accum, original_shape)

def gaussian_blur(img, sigma):
    '''A Gaussian blur (also known as Gaussian smoothing) is the result of blurring an image by a Gaussian function.
    ----------
    Parameters
    ----------
        img: np.array . Multi-dimensional numpy array in format (height, width, n_channels)
    '''
    from scipy.ndimage.filters import gaussian_filter
    import concurrent.futures
    def blur_channel(channel, sigma):
        return gaussian_filter(channel, sigma=sigma)

    time_script_start = timer()
    # Use a ThreadPoolExecutor to apply the Gaussian filter to each channel in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        blurred_channels = list(executor.map(blur_channel, [img[:,:,i] for i in range(img.shape[-1])], [sigma]*img.shape[-1]))
    
    # Concatenate the blurred channels back into a single image
    blurred = np.array(blurred_channels)
    blurred = np.moveaxis(blurred, 0, -1)
    time_script_end = timer()
    logger.debug(
        "--Gaussian bluring execution time:\t%2.2f minutes"% np.round((time_script_end - time_script_start) / 60, 2),2)

    return blurred

@jit
def calculate_vi(img):
    '''Calculates ancillary remote-sensing vegetation indices.
    ----------
    Parameters
    ----------
    img: np.array . Multi-dimensional numpy array in format (height, width, n_channels)
    '''
    time_script_start = timer()
    green_band = img[...,1]
    red_band = img[...,2]
    nir_band = img[...,6]

    #Caclulate Normalized Difference Vegetation Index (NDVI)
    ndvi = (nir_band - red_band) / (nir_band + red_band)

    #Optimized Soil Adjusted Vegetation Index (OSAVI)
    osavi = (nir_band - red_band) / (nir_band + red_band + 0.16)

    #Soil Adjusted Vegetation Index (SAVI)
    savi = 1.5*(nir_band - red_band) / (nir_band + red_band + 0.5)

    # Calculate GNDVI
    gndvi = (nir_band - green_band) / (nir_band + green_band)

    # Calculate NDWI
    ndwi = (green_band - nir_band) / (green_band + nir_band)

    #ndvi, osavi, savi = norm_values(ndvi), norm_values(osavi), norm_values(savi)
    vis = np.stack((ndvi,osavi,savi, gndvi, ndwi),axis=2)
    vis = np.nan_to_num(vis, nan=0)

    time_script_end = timer()
    logger.debug(
        "--Calculation of VIs execution time:\t%2.2f minutes"% np.round((time_script_end - time_script_start) / 60, 2),2)
    return vis

def run_kmeans_faiss(fs, num_clusters):
    '''Performs the kmeans clustering, which k-means clustering is a method of vector quantization, 
    originally from signal processing, that aims to partition n observations into k clusters in which 
    each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.
    ----------
    Parameters
    ----------
        fs: numpy.array. 2-dimensional array, 1st_dimension: rows*columns of image, 2nd_dimension: n_channels
        num_clusters: int. The desired number of clusters.
    '''

    import faiss

    #Remove nans
    fs = np.nan_to_num(fs) 

    #Run PCA
    logger.info("---Starting PCA")
    time_script_start = timer()    
    pca = faiss.PCAMatrix(fs.shape[1], 10)
    fs = fs.astype(np.float32)
    fs = np.ascontiguousarray(fs)
    pca.train(fs)
    principalComponents = pca.apply(fs)
    time_script_end = timer()
    logger.info("---PCA finished successfully.")
    logger.debug("---PCA execution time:\t%2.2f minutes"% np.round((time_script_end - time_script_start) / 60, 2),2)

    #Run K-means
    logger.info(f"---Starting k-means with number of clusters {num_clusters}")
    time_script_start = timer()   
    nredo = 150 
    niter = 100
    verbose = True
    fs = fs.astype(np.float32)
    d = principalComponents.shape[1]
    # d = fs.shape[1]

    kmeans = faiss.Kmeans(d, num_clusters, 
                             niter=niter, 
                             verbose=verbose,
                             nredo=nredo,
                             update_index=True,
                             seed=123)

    kmeans.train(principalComponents) #principalComponents, fs
    result = kmeans.index.search(x=principalComponents, k=1)[1].reshape(-1) #principalComponents, fs
    time_script_end = timer()
    logger.info(f"---K-means prediction finished")
    logger.debug("---K-means execution time:\t%2.2f minutes"% np.round((time_script_end - time_script_start) / 60, 2),2)

    return result

def calculate_zonal_stats_multi(img, mask):
    '''Calculate the zonal statistics of the image values based on
    the calculated clusters from k-means. It returns a dataframe of 
    the calculated zonal statistics.
    ----------
    Parameters
    ----------
    img: np.array. Multi-dimensional numpy array in format (height, width, n_channels)
    mask: np.array. Two-dimensional clustering map produced by k-means (height, width)
    '''
    # Import necessary modules
    import pandas as pd
    import threading
    time_script_start = timer()

    # Define the zonal_stats function
    def zonal_stats(arr, val):
        '''Calculation of max, min, mean, median, std and the majority value 
        per unique cluster from k-means.
        ----------
        Parameters
        ----------
        arr: np.array. One-dimensional numpy array, refers to pixel values which belong 
                       to a unique cluster class.
        '''
        from numpy import amax, amin, mean, median, std
        
        maxim, minim = amax(arr), amin(arr)
        mean, median = mean(arr), median(arr)
        stdin = std(arr)
        unique, counts = np.unique(arr, return_counts=True)
        # Find the element with the highest count
        majority = unique[np.argmax(counts)]

        results = [maxim, minim, mean, median, stdin, majority, val]
        return results          

    def sample_data(data, sample_fraction):
        '''Sampling down the data to minimize the execution time of the followed zonal statistics.
        ----------
        Parameters
        ----------
        data: np.array. One-dimensional numpy array, refers to pixel values.
        sample_fraction: int. The fraction of the total size.
        '''
        import numpy as np
        # Set the random seed for reproducibility
        np.random.seed(42)
        #logger.info(f"--Now sampling the data for the cluster")
        n_samples = int(data.shape[0] * sample_fraction)  # Number of samples to take
        sample_indices = np.random.choice(data.shape[0], size=n_samples, replace=False)  # Indices of the samples
        sample = data[sample_indices]  # Subset of the data

        return sample  

    def calculate_stats_for_zone(cluster, img, mask, zn_stats):
        logger.info(f"--Now calculating the zonal statistics for cluster: {cluster}")
        # Initialize a list to store the zonal statistics for the current zone
        stats_for_zone = []
        # Iterate over each layer of the image
        for m in range(img.shape[-1]):
            # Compute the zonal statistics for the current layer and zone
            img_ar = img[...,m]
            arr = img_ar[mask==cluster]
            # Get sample from the whole dataset to speed up the calculation of the zonal statistics
            sample = sample_data(arr, 0.3)
            # Calculate the zonal statistics
            stats = zonal_stats(sample, cluster)
            # Append the zonal statistics to the list
            stats_for_zone.append(stats)
        df = pd.DataFrame(stats_for_zone)
        zn_stats.append(df)
        return zn_stats

    # Get the unique values in the mask array
    uniq = np.unique(mask)
    logger.info(f"--- K-means resulted in {len(uniq)} classes")
    zn_stats = []
    # create a thread for each cluster
    threads = []
    for cluster in uniq:
        thread = threading.Thread(target=calculate_stats_for_zone, args=(cluster, img, mask, zn_stats))
        threads.append(thread)
        thread.start()

    # wait for all threads to finish
    for thread in threads:
        thread.join()

    # Convert the zn_stats array to a Pandas DataFrame
    df = pd.concat(zn_stats)
    df.columns = ['max', 'min', 'mean', 'median', 'std', 'majority', 'class']

    # Create the list l using a list comprehension
    l = [[f'Band {m}'] * (len(uniq)) for m in range(1, img.shape[-1]+1)]

    # Flatten the list l into a single list
    fl = [item for sublist in l for item in sublist]
    df['band'] = fl
    time_script_end = timer()
    logger.debug("---Zonal statistics execution time:\t%2.2f minutes"% np.round((time_script_end - time_script_start) / 60, 2),2)
    return df

def main():
    #Select one image
    path = "./path2multibands_rasters/multibands"
    imgpath = "20230306T031619_mutliband.jp2"
    image = os.path.join(path, imgpath)
    logger.info("Starting k-means segmentation pipeline")

    #1. Load image
    logger.info(f"--1. Loading image: {image.split('/',-1)[-1]}")
    with rasterio.open(image) as src:
            data = src.read()
            profile = src.profile
            nodata = profile['nodata']
            src.close()

    #2. Reshape image from [n_channels x height x width] to [height x width x n_channels]
    logger.info("--2. Preparing features of the given image")
    img = np.moveaxis(data, 0, -1)

    #3. Apply gaussian bluring to reduce noise
    logger.info("--3. Applying gaussian bluring")
    img = gaussian_blur(img, sigma=1.5)

    #4. Apply gabor filtering
    logger.info("--4. Apply gabor filtering")
    textimg = gabor_process_threaded(img, 0.01, 64)

    #5. Calculate vis
    logger.info("--5. Calculate Vegetation Indices")
    visimg = calculate_vi(img)

    # Merge output with image
    img = np.concatenate((img,textimg,visimg),axis=2)#textimg

    #6. Create Feature Space by flattening heightxwidth
    logger.info("--6. Creation of Feature Space")
    fs = img.reshape(-1,img.shape[-1])

    #7. Run clustering by applying PCA to reduce feature space from 4 to 2
    # and then run k-means
    num_clusters = 20
    output = f"/efs/prototyping/antonios/laos_deforestation/kmeans/{imgpath[:-4]}_kmeans_zonal.tif"

    logger.info("--7. Starting clustering")
    result = run_kmeans_faiss(fs , num_clusters)

    #8. Reshape prediction image back to the original dimensions
    logger.info("--8. Reshaping prediction image back to the original dimensions")
    result = result.reshape(data.shape[1],data.shape[2])
    result = result.astype(np.float32)
    new_nodata = np.nan
    result[np.isnan(data[0])] = new_nodata

    #9. Write result to output path
    logger.info("--9. Writting result to output path")
    profile.update(driver='GTiff', 
                    dtype=rasterio.float32,
                    count=1,
                    nodata=new_nodata,)
    
    with rasterio.open(output, 'w', **profile) as dst:
        dst.write(result, 1)
    
if __name__ == "__main__":
    time_script_start = timer()
    main()
    time_script_end = timer()
    logger.debug(
        "Script execution time:\t%2.2f minutes"% np.round((time_script_end - time_script_start) / 60, 2),2)
