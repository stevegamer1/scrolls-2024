import tifffile
import cv2 as cv
import cv2.ximgproc as ximgproc
import numpy as np


def fill_small_black_regions(bw_image):
    """
        Fill all regions of 0 with 255 except the biggest one.
    """
    # label black regions
    count, labeled, stats, centroids = cv.connectedComponentsWithStats(255 - bw_image, connectivity=4)
    max_area = 0
    for i in range(count)[1:]:
        max_area = max(max_area, stats[i][4])
    max_label = None
    for i in range(count)[1:]:
        area = stats[i][4]
        if area == max_area:
            max_label = i
    
    result = np.logical_not(np.isin(labeled, [max_label])) * np.ones_like(labeled, np.uint8) * 255
    return result


def smooth_from_canny(image_after_canny, starting_data):
    image_after_canny = cv.dilate(image_after_canny, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    image_after_canny = ximgproc.thinning(image_after_canny)

    # 1. remove small components
    count, labeled, stats, centroids = cv.connectedComponentsWithStats(image_after_canny)
    big_labels = set()
    for i in range(count)[1:]:
        area = stats[i][4]
        if area >= 200:
            big_labels.add(i)
    canny_clean = np.isin(labeled, list(big_labels)) * np.ones_like(labeled, np.uint8)
    #tifffile.imwrite("canny_without_trash.tif", 255 * canny_clean)

    # 2. dilate and thin
    dilated = cv.dilate(255 * canny_clean, cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4)))
    thinned = ximgproc.thinning(dilated)
    #tifffile.imwrite("canny_dilated_thinned.tif", thinned)
    #tifffile.imwrite("clean_edges_with_original.tif", np.array(thinned / 255, np.float32) + (starting_data - np.min(starting_data)) / (np.max(starting_data) - np.min(starting_data)))

    # 3. fill the inside
    filled = fill_small_black_regions(thinned)
    #tifffile.imwrite("canny_filled.tif", filled)
    return filled


def canny_based_segmentation(data: np.ndarray):
    #canny_input = (data > 0.00075) * data
    canny_input = data
    sigma = 5.0
    canny_input = cv.GaussianBlur(canny_input, (int(sigma * 3.0), int(sigma * 3.0)), sigmaX=sigma, sigmaY=sigma)
    canny_input = np.array((canny_input - np.min(canny_input)) / np.max(canny_input) * 256.0, np.uint8)
    canny = cv.Canny(canny_input, 0.1, 10.0)
    smooth_filled = smooth_from_canny(canny, data)
    return smooth_filled


def demonstrate_canny_based():
    filename = "../scroll001.rec_1150/SPTV0028.tif"
    data = tifffile.imread(files=filename)
    tifffile.imwrite("canny_based_result.tif", canny_based_segmentation(data))

