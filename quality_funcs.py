import nrrd
import os
import config
import main
import numpy as np
import tifffile
import itertools
import scipy.signal as signal
import scipy.spatial.distance as distance
import functools
import cv2 as cv
import canny_experiments
import trimesh


def calc_IoU(arr1, arr2):
    assert(arr1.dtype == bool)
    assert(arr2.dtype == bool)
    assert(arr1.shape == arr2.shape)
    intersection = arr1 * arr2
    union = arr1 + arr2
    return intersection.sum() / float(union.sum())


def calc_relative_area_diff(arr1, arr2):
    assert(arr1.dtype == bool)
    assert(arr2.dtype == bool)
    assert(arr1.shape == arr2.shape)
    area1 = float(arr1.sum())
    area2 = float(arr2.sum())
    return (area1 - area2) / area2


def surface_distance_matrix(arr1_bool, arr2_bool):
    """
        Distances between all surface (8-neighborhood) points of regions.
    """
    assert(arr1_bool.dtype == bool)
    assert(arr2_bool.dtype == bool)
    arr1 = np.array(arr1_bool, np.int8)
    arr2 = np.array(arr2_bool, np.int8)
    assert(arr1.shape == arr2.shape)

    surface_kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], np.int8)

    arr1_surface = np.column_stack(np.nonzero(np.logical_and(arr1, signal.convolve2d(1 - arr1, surface_kernel, mode='same'))))
    arr2_surface = np.column_stack(np.nonzero(np.logical_and(arr2, signal.convolve2d(1 - arr2, surface_kernel, mode='same'))))
    distances = distance.cdist(arr1_surface, arr2_surface)
    return distances, arr1_surface, arr2_surface


def calc_ASD(distances_matrix, arr1_surface, arr2_surface):
    """
        Average Symmetric Distance. The higher it is, the more different are the arrays.
    """
    arr1_to_arr2 = np.sum(np.min(distances_matrix, axis=1))
    arr2_to_arr1 = np.sum(np.min(distances_matrix, axis=0))
    result = (arr1_to_arr2 + arr2_to_arr1) / (arr1_surface.shape[0] + arr2_surface.shape[0])
    return result


def calc_RMSD(distances_matrix, arr1_surface, arr2_surface):
    """
        Root Mean Symmetric Distance. The higher it is, the more different are the arrays. It punishes outliers more than ASD.
    """
    arr1_to_arr2 = np.min(distances_matrix, axis=1)
    arr1_to_arr2 = np.dot(arr1_to_arr2, arr1_to_arr2)
    arr2_to_arr1 = np.min(distances_matrix, axis=0)
    arr2_to_arr1 = np.dot(arr2_to_arr1, arr2_to_arr1)
    result = np.sqrt((arr1_to_arr2 + arr2_to_arr1) / (arr1_surface.shape[0] + arr2_surface.shape[0]))
    return result


def calc_Hausdorff(distances_matrix, arr1_surface, arr2_surface):
    """
        Maximum Symmetric Distance. If it's higher, then somewhere there is a bigger difference between the arrays.
    """
    arr1_to_arr2 = np.max(np.min(distances_matrix, axis=1))
    arr2_to_arr1 = np.max(np.min(distances_matrix, axis=0))
    result = max(arr1_to_arr2, arr2_to_arr1)
    return result


def calc_Hausdorff95(distances_matrix, arr1_surface, arr2_surface):
    """
        Like Hausdorff's distance, but 95th percentile is taken instead of maximum. Less sensitive to outliers.
    """
    arr1_to_arr2 = np.percentile(np.min(distances_matrix, axis=1), 95)
    arr2_to_arr1 = np.percentile(np.min(distances_matrix, axis=0), 95)
    result = max(arr1_to_arr2, arr2_to_arr1)
    return result


example_arr1 = np.array([[1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]], bool)
example_arr2 = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1],
                         [0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 1]], bool)

diff_dist_mat, example_arr1_surf, example_arr2_surf = surface_distance_matrix(example_arr1, example_arr2)
same_dist_mat, _, _ = surface_distance_matrix(example_arr1, example_arr1)
assert(abs(calc_ASD(diff_dist_mat, example_arr1_surf, example_arr2_surf) - 1.128564) < 0.0001)
assert(calc_ASD(same_dist_mat, example_arr1_surf, example_arr1) == 0)
assert(calc_RMSD(same_dist_mat, example_arr1_surf, example_arr1_surf) == 0)
assert(calc_Hausdorff(same_dist_mat, example_arr1_surf, example_arr1_surf) == 0)
assert(calc_Hausdorff95(same_dist_mat, example_arr1_surf, example_arr1_surf) == 0)


assert(calc_IoU(np.array([True], bool), np.array([True], bool)) == 1.0)
assert(calc_IoU(np.array([True], bool), np.array([False], bool)) == 0.0)
assert(calc_IoU(np.array([False], bool), np.array([True], bool)) == 0.0)
assert(calc_IoU(np.array([False, True], bool), np.array([True, True], bool)) == 0.5)
assert(calc_IoU(np.array([False, True, True], bool), np.array([True, True, False], bool)) == (1.0 / 3.0))


assert(calc_relative_area_diff(np.array([True], bool), np.array([True], bool)) == 0.0)
assert(calc_relative_area_diff(np.array([False], bool), np.array([True], bool)) == -1.0)
assert(calc_relative_area_diff(np.array([True, True], bool), np.array([True, False], bool)) == 1.0)



def get_ground_truth():
    # ground_truth, header = nrrd.read(os.path.abspath("../ground_truth/scroll001preview_slicerfiles/Segmentation.seg.nrrd"))
    ground_truth, header = nrrd.read(os.path.abspath("../ground_truth/scroll001full_slicerfiles/segm_ground_truth/fullvolume001.seg.nrrd"))
    ground_truth = np.swapaxes(ground_truth, 0, 2)
    return ground_truth


def calc_quality_getlinefor():
    """
        This function calculates quality functions for get_line_for algorithm.
    """
    ground_truth = get_ground_truth()

    IoUs = []
    relative_area_diffs = []
    ASDs = []
    RMSDs = []
    Hausdorffs = []
    Hausdorffs95 = []
    
    group_len = 1  # how many files to take at each of the evenly distributed places
    group_interval = 400  # step at which the places are distributed evenly
    groups = [config.filenames[start:start + group_len] for start in range(2, len(config.filenames), group_interval)]
    chosen_files = list(itertools.chain.from_iterable(groups))
    for i in range(len(chosen_files)):
        print("Quality functions calculation progress:", i / len(chosen_files) * 100, "%")
        
        line, with_line = main.get_line_for(chosen_files[i], True)
        with_line_bool = with_line > 0
        grtruth_bool = ground_truth[i] > 0

        IoUs.append(calc_IoU(with_line_bool, grtruth_bool))
        relative_area_diffs.append(calc_relative_area_diff(with_line_bool, grtruth_bool))
        distance_matrix, withline_surface, grtruth_surface = surface_distance_matrix(with_line_bool, grtruth_bool)
        ASDs.append(calc_ASD(distance_matrix, withline_surface, grtruth_surface))
        RMSDs.append(calc_RMSD(distance_matrix, withline_surface, grtruth_surface))
        Hausdorffs.append(calc_Hausdorff(distance_matrix, withline_surface, grtruth_surface))
        Hausdorffs95.append(calc_Hausdorff95(distance_matrix, withline_surface, grtruth_surface))
    
    return chosen_files, IoUs, relative_area_diffs, ASDs, RMSDs, Hausdorffs, Hausdorffs95


def assess_getlinefor():
    """
        Calculate quality functions for get_line_for, write them to file, print the results.
    """
    chosen_files, IoUs, RADs, ASDs, RMSDs, Hausdorffs, Hausdorffs95 = calc_quality_getlinefor()

    print("Mean IoU: ", np.mean(IoUs))
    print("Mean relative area difference: ", np.mean(RADs))
    print("Mean absolute surface distance: ", np.mean(ASDs))
    print("Mean root mean square distance: ", np.mean(RMSDs))
    print("Mean Haussdorff's distance: ", np.mean(Hausdorffs))
    print("Mean Haussdorff's-95 distance: ", np.mean(Hausdorffs95))

    with open("getlinefor_results.txt", 'w') as f:
        f.write("Chosen files: " + str(chosen_files) + "\n")
        f.write("IoUs: " + str(IoUs) + "\n")
        f.write("RADs: " + str(RADs) + "\n")
        f.write("ASDs: " + str(ASDs) + "\n")
        f.write("RMSDs: " + str(RMSDs) + "\n")
        f.write("Hausdorffs: " + str(Hausdorffs) + "\n")
        f.write("Hausdorffs-95: " + str(Hausdorffs95) + "\n")


def do_quality_functions_experiments(output_filename: str):
    """
        Put ground truth and it's altered versions in quality functions and see what the functions say.
    """
    ground_truth = get_ground_truth()

    chosen_slices = ground_truth[range(0, len(ground_truth), 400)]
    chosen_slices = [chosen_slices[2]]

    def noise_border(arr: np.ndarray, p: float):
        """
            Flip random p * 100% of pixels, but only choose pixels on a thick border.
        """
        dilated = cv.dilate(np.array(arr, np.uint8), cv.getStructuringElement(cv.MORPH_RECT, (4, 4)))
        eroded = cv.erode(np.array(arr, np.uint8), cv.getStructuringElement(cv.MORPH_RECT, (4, 4)))
        border_mask = (dilated + eroded) % 2  # xor
        rng = np.random.default_rng(42)
        noise = np.logical_and(border_mask, np.array(rng.binomial(1, p, border_mask.shape), bool))
        tifffile.imwrite("noised.tif", np.array(np.logical_xor(arr, noise), np.uint8) * 255)
        return np.logical_xor(arr, noise)

    def erode_bool(arr, k_size):
        return np.array(cv.erode(np.array(arr, np.uint8), cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))), bool)

    def dilate_bool(arr, k_size):
        return np.array(cv.dilate(np.array(arr, np.uint8), cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))), bool)

    experiments = {
        "Shift-X-1" : functools.partial(np.roll, shift=1, axis=1),
        "Shift-X-10" : functools.partial(np.roll, shift=10, axis=1),
        "Noise 5%, border:" : functools.partial(noise_border, p=0.05),
        "Erode 3 pixels:" : functools.partial(erode_bool, k_size=3),
        "Dilate 3 pixels:" : functools.partial(dilate_bool, k_size=3)
    }
    
    results = {}
    for exp_name in experiments.keys():
        results[exp_name] = {"IoUs":[], "RADs":[], "ASDs":[], "RMSDs":[], "Hausdorffs":[], "Hausdorffs-95":[]}
    
    for slice_float, i in zip(chosen_slices, range(len(chosen_slices))):
        print("progress", i / len(chosen_slices) * 100, "%")

        slice = slice_float > 0
        for exp_name, experiment in experiments.items():
            exp_slice = experiment(slice)
            results[exp_name]["IoUs"].append(calc_IoU(slice, exp_slice))
            results[exp_name]["RADs"].append(calc_relative_area_diff(slice, exp_slice))
            dist_mat, slice_surf, exp_surf = surface_distance_matrix(slice, exp_slice)
            results[exp_name]["ASDs"].append(calc_ASD(dist_mat, slice_surf, exp_surf))
            results[exp_name]["RMSDs"].append(calc_RMSD(dist_mat, slice_surf, exp_surf))
            results[exp_name]["Hausdorffs"].append(calc_Hausdorff(dist_mat, slice_surf, exp_surf))
            results[exp_name]["Hausdorffs-95"].append(calc_Hausdorff95(dist_mat, slice_surf, exp_surf))
    
    for exp_name, experiment in experiments.items():
        print("Experiment: " + exp_name)
        for key, value in results[exp_name].items():
            print(key + " mean:", np.mean(value))
        print()

    with open(output_filename, 'w') as f:
        for exp_name, experiment in experiments.items():
            f.write("Experiment: " + exp_name + "\n")
            for key, value in results[exp_name].items():
                f.write(key + ": " + str(value) + "\n")
            f.write("\n")


def assess_algorithms(output_filename: str):
    def smooth1(slice: np.ndarray, slice_index: int):
        return main.smooth1(slice,
            denoise_threshold=config.smooth1_configs['denoise_threshold'],
            aftergauss_threshold=config.smooth1_configs['aftergauss_threshold'],
            gauss_sigma=config.smooth1_configs['gauss_sigma'],
            erosion_size=config.smooth1_configs['erosion_size'])

    def canny_based(slice: np.ndarray, slice_index: int):
        return canny_experiments.canny_based_segmentation(slice)

    mesh_raw = main.make_mesh()
    mesh = trimesh.Trimesh(vertices=mesh_raw[0], faces=mesh_raw[1])
    def segmentation_from_mesh(slice: np.ndarray, slice_index: int):
        section = mesh.section(plane_origin=[0, 0, slice_index], plane_normal=[0, 0, 1])
        section.explode()
        result = np.zeros(slice.shape, np.int8)
        for i in range(len(section.vertices)):
            start = np.array(section.vertices[section.entities[i].end_points[0]][:2], np.int32)
            end = np.array(section.vertices[section.entities[i].end_points[1]][:2], np.int32)
            result = cv.line(result, start, end, 255, 13)
        return result

    algorithms = {
        "smooth1" : smooth1,
        "Canny filter-based" : canny_based,
        "make mesh & dilate" : segmentation_from_mesh
    }

    ground_truth = get_ground_truth()

    results = {}
    for algo_name, algorithm in algorithms.items():
        results[algo_name] = {"IoUs":[], "RADs":[], "ASDs":[], "RMSDs":[], "Hausdorff's 95":[], "Hausdorff's":[]}

    chosen_slices = [800]
    for slice_i in chosen_slices:
        grtruth_slice = ground_truth[slice_i] > 0
        tifffile.imwrite("ground truth.tif", np.array(grtruth_slice, np.uint8) * 255)
        data_slice = tifffile.imread(config.filenames[slice_i])
        for algo_name, algorithm in algorithms.items():
            segmentation = np.array(algorithm(data_slice, slice_i), bool)
            tifffile.imwrite(algo_name + " result.tif", np.array(segmentation, np.uint8) * 255)
            results[algo_name]["IoUs"].append(calc_IoU(grtruth_slice, segmentation))
            results[algo_name]["RADs"].append(calc_relative_area_diff(grtruth_slice, segmentation))
            dist_mat, grtruth_surf, segm_surf = surface_distance_matrix(grtruth_slice, segmentation)
            results[algo_name]["ASDs"].append(calc_ASD(dist_mat, grtruth_surf, segm_surf))
            results[algo_name]["RMSDs"].append(calc_RMSD(dist_mat, grtruth_surf, segm_surf))
            results[algo_name]["Hausdorff's 95"].append(calc_Hausdorff95(dist_mat, grtruth_surf, segm_surf))
            results[algo_name]["Hausdorff's"].append(calc_Hausdorff(dist_mat, grtruth_surf, segm_surf))

    for algo_name, algorithm in algorithms.items():
        print("Algorithm: " + algo_name)
        for func_name, result in results[algo_name].items():
            print(func_name + " mean:", np.mean(result))
        print()
    
    with open(output_filename, 'w') as f:
        for algo_name, algorithm in algorithms.items():
            f.write("Algorithm: " + algo_name + "\n")
            for func_name, result in results[algo_name].items():
                f.write(func_name + ": " + str(result) + "\n")
            f.write("\n")


assess_algorithms("algorithms results.txt")
